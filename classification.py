import numpy as np
import cv2
import scipy.io
from dataclasses import dataclass
import os
import copy
from generateClassificationImages import generate_classification_images
from generateClassificationImages import outputImStruct
import datetime
import mat73
from netOnly4 import netOnly4
from netOnly7 import netOnly7
from netOnly1 import netOnly1
import multiprocessing as mp
from joblib import Parallel, delayed
import traceback

@dataclass
class fieldStruct:
    ID: int = None
    pixIndex: np.ndarray = None
    trueValue: int = None
    pixCount: int = None

@dataclass
class netType:
    nanLocations: np.ndarray = None
    cropTypes: np.ndarray = None
    notCrop: np.ndarray = None
    numClasses: int = None
    results_pix: np.ndarray = None
    results_field: np.ndarray = None
    accuracy_pix: np.ndarray = None
    accuracy_field: np.ndarray = None
    pixelClassData: np.ndarray = None
    done: bool = None
    isStart: bool = None
    outputIm: outputImStruct = outputImStruct()

def initialize_net_type(nanLocations, cropTypes, pixCount, DataList):
    netTypeObj = netType()
    netTypeObj.nanLocations = np.array(nanLocations)
    netTypeObj.cropTypes = np.array(cropTypes)
    netTypeObj.notCrop = np.zeros((1, pixCount))
    netTypeObj.numClasses = len(netTypeObj.cropTypes)
    netTypeObj.results_pix = np.zeros((netTypeObj.numClasses, netTypeObj.numClasses, DataList.shape[1-1]))
    netTypeObj.results_field = np.zeros((netTypeObj.numClasses, netTypeObj.numClasses, DataList.shape[1-1]))
    netTypeObj.accuracy_pix = np.zeros((1, DataList.shape[1-1]))
    netTypeObj.accuracy_field = np.zeros((1, DataList.shape[1-1]))
    netTypeObj.pixelClassData = np.ones((1, pixCount)) * -1
    netTypeObj.done = 0
    netTypeObj.isStart = 0
    return netTypeObj

## loop to genereate results
def process(i,showIm,imInd,isFieldLevelOnly,minMaturity,
            maxMaturity, startMaturity,minAvgMaurity,
            minFractionForDecision,fieldCount,sensorNames,
            fixedCrop,evalDayArray,cropList,results_field,accuracy_field,
            netTypeArray,outputIm,currState, currCovar, currDateGDD,accuLikeli,
            accuCount, totalAccu,DataList,DataFolder,netTypeCount,sensorTypeCount,
            cropTypeCount,pixelClassData,fieldStructArray,rowI,colJ,maskImage,anyUpdate ):
    
    currState = currState.copy()
    currCovar = currCovar.copy()
    currDateGDD = currDateGDD.copy()
    totalAccu = totalAccu.copy()
    accuCount = accuCount.copy()
    accuLikeli = accuLikeli.copy()
    pixelClassData = pixelClassData.copy()

    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def show_image(image, title="image"):
            resize = ResizeWithAspectRatio(image / 255, width=1900)
            cv2.imshow(title, resize)
            cv2.waitKey(0)
            cv2.destroyWindow(title)
    
    print(f"Processing file {i} in process {os.getpid()}")
    print('Reading image data {}\n'.format(i))
    currentFileName = DataFolder + '\\' + DataList[i]
    currentFile = mat73.loadmat(currentFileName)
    likeliData1 = currentFile['likeliData1']
    evalDay = likeliData1[0, 3]
    evalDayArray.append(evalDay)
    sensorInd = int(likeliData1[0,5]) - 1
    datestr = datetime.date.fromordinal(int(evalDay)) + datetime.timedelta(days=int(evalDay)%1) - datetime.timedelta(days = 366)
    print('Data on {0} from {1}\n'.format(str(datestr), sensorNames[sensorInd]))
    ## determine classification type
    for n in range(0, netTypeCount):
        skipIt = 0
        for p in range(0, len(netTypeArray[n].nanLocations)):
            I = np.squeeze(np.where(np.logical_and(np.invert(np.isnan(likeliData1[:,netTypeArray[n].nanLocations[p]])), likeliData1[:,netTypeArray[n].nanLocations[p]]) > 0))
            if (np.size(I)==0):
                skipIt = 1
                break
            if (np.median(likeliData1[I,netTypeArray[n].nanLocations[p]]) < minMaturity or np.median(likeliData1[I,netTypeArray[n].nanLocations[p]]) > maxMaturity[netTypeArray[n].cropTypes[p]]):
                skipIt = 1
                if (np.median(likeliData1[I,netTypeArray[n].nanLocations[p]]) > maxMaturity[netTypeArray[n].cropTypes[p]]):
                    netTypeArray[n].done = 1
                break
        if (not skipIt  and not netTypeArray[n].done):
            I = np.squeeze(np.where(np.logical_and(np.logical_and(np.all(np.invert(np.isnan(likeliData1[:,netTypeArray[n].nanLocations])), 1), np.all(likeliData1[:,netTypeArray[n].nanLocations] > 0, 1)),
                                                   np.all(np.invert(np.isinf(likeliData1[:, 8:18:3])), 1))))
            pixIndex = likeliData1[I,0].astype(np.int64) - 1
            currState[pixIndex,:] = copy.copy(likeliData1[I, 6:-1:3])
            currCovar[pixIndex,:] = copy.copy(likeliData1[I, 7:-1:3])
            currDateGDD[pixIndex,:] = copy.copy(likeliData1[I, 3:5])
            totalAccu[pixIndex,n] = copy.copy(totalAccu[pixIndex,n]) + 1
            accuCount[pixIndex,(n - 1) * sensorTypeCount + sensorTypeCount + sensorInd] = accuCount[pixIndex,(n - 1) * sensorTypeCount + sensorTypeCount + sensorInd] + 1
            accuLikeli[pixIndex[:,None],((n - 1) * cropTypeCount + cropTypeCount + netTypeArray[n].cropTypes[0:-1])[None,:]] = accuLikeli[pixIndex[:,None],((n - 1) * cropTypeCount + cropTypeCount + netTypeArray[n].cropTypes[0:-1])[None,:]] + likeliData1[I[:,None], (9 + (netTypeArray[n].cropTypes[0:-1] - 1) * 3 + 2)[None,:]]
            pixelClassData[:, 2] = -1
            validPix = totalAccu[:, n] > 0
            if validPix.sum() > 1:
                anyUpdate = 1
                I = np.squeeze(np.where(validPix))
                #build state and likelihood--
                stateLikeliMat = np.zeros((len(I),(netTypeArray[n].numClasses - 1) * 3))
                for k in range(0, netTypeArray[n].numClasses - 1):
                    crop_type_k = netTypeArray[n].cropTypes[k]
                    stateLikeliMat[:, (k * 3):(k * 3 + 3)] = np.stack((currState[I, crop_type_k], 
                                                                       currCovar[I, crop_type_k], 
                                                                       accuLikeli[I,(n - 1) * cropTypeCount + cropTypeCount + crop_type_k] / totalAccu[I,n]), axis=-1)
                accu_ratio = accuCount[I].T[(n - 1) * sensorTypeCount + sensorTypeCount + (np.arange(sensorTypeCount))].T / np.repeat(totalAccu[I,n][:, None], 3, axis=1)
                testData = np.concatenate((currDateGDD[I,1][:, None], totalAccu[I,n][:, None], accu_ratio, stateLikeliMat), axis=1)
                testData = testData if n == 1 else testData[:, :8]  # 8xQ matrix for n!=1 and 11xQ matrix for n=1
                net_func = [netOnly4, netOnly7, netOnly1][n]
                resData = np.round(net_func(testData.T)).astype(np.byte)
                # Mapping of resData based on n
                resData_map = {0: {1: 0, 2: 4},
                               1: {1: 1, 2: 2, 3: 4},
                               2: {1: 3, 2: 4}}
                for k, v in resData_map[n].items():
                    resData[resData == k] = v
                pixelClassData[I,2] = resData
                n_crop_types = len(netTypeArray[n].cropTypes)
                crop_types = netTypeArray[n].cropTypes
                # Compute accuracy
                confusionMat = np.zeros((n_crop_types, n_crop_types))
                correctNum = 0
                # Create mapping for pixel class data
                pixel_class_data_map = {p: np.where(pixelClassData[:, 1] == crop_types[p])[0] for p in range(n_crop_types - 1)}
                for p in range(n_crop_types - 1):
                    trueI = pixel_class_data_map[p]
                    trueI_len = len(trueI)
                    confusionMat[p, :n_crop_types - 1] = [np.sum(pixelClassData[trueI, 2] == crop_types[q]) / trueI_len for q in range(n_crop_types - 1)]
                    correctNum += np.sum(np.eye(n_crop_types - 1)[p] * confusionMat[p, :n_crop_types - 1]) * trueI_len
                    confusionMat[p, n_crop_types - 1] = 1 - np.sum(confusionMat[p, :n_crop_types - 1])
                trueI = np.where(~np.isin(pixelClassData[:, 1], crop_types[:n_crop_types - 1]))[0]
                trueI_len = len(trueI)
                confusionMat[-1, :n_crop_types - 1] = [np.sum(pixelClassData[trueI, 2] == crop_types[q]) / trueI_len for q in range(n_crop_types - 1)]
                confusionMat[-1, -1] = 1 - np.sum(confusionMat[-1, :n_crop_types - 1])
                correctNum += confusionMat[-1, -1] * trueI_len
                accuracy = correctNum / len(imInd)
                print("confusionMat:")
                print(confusionMat)
                print("accuracy: ", str(accuracy))
                netTypeArray[n].results_pix[:, :, i] = confusionMat.copy()
                netTypeArray[n].accuracy_pix[0][i] = accuracy
                # Fix some data
                # Find if maturity is enough
                isMature = isStart = all(np.mean(stateLikeliMat[np.where(~np.isnan(stateLikeliMat[:, i])), i]) >= minAvgMaurity
                                          for i in range(1, (len(netTypeArray[n].cropTypes) - 1) * 3, 3))
                isStart = isStart and all(np.mean(stateLikeliMat[np.where(~np.isnan(stateLikeliMat[:, i])), i]) >= startMaturity
                                          for i in range(1, (len(netTypeArray[n].cropTypes) - 1) * 3, 3) if not netTypeArray[n].isStart)
                netTypeArray[n].isStart = isStart
                if isMature:
                    for f in range(fieldCount):
                        if not (any(fixedCrop[:, fieldStructArray[f].pixIndex[0]] == 1) or all(netTypeArray[n].notCrop[fieldStructArray[f].pixIndex[0]] == 1)):
                            for p in range(len(netTypeArray[n].cropTypes) - 1):
                                if (pixelClassData[fieldStructArray[f].pixIndex, 2] == netTypeArray[n].cropTypes[p]).sum() / fieldStructArray[f].pixCount > minFractionForDecision:
                                    fixedCrop[netTypeArray[n].cropTypes[p], fieldStructArray[f].pixIndex] = 1
                            if (pixelClassData[fieldStructArray[f].pixIndex, 2] == netTypeArray[n].cropTypes[-1]).sum() / fieldStructArray[f].pixCount > minFractionForDecision:
                                netTypeArray[n].notCrop[fieldStructArray[f].pixIndex] = 1
               # Keep the previously fixed data
                pixelClassData[netTypeArray[n].notCrop.squeeze() == 1, 2] = 4
                # Make the changes here
                for p in range(cropTypeCount):
                    mask = fixedCrop[p, :] == 1
                    pixelClassData[mask, 2] = p
                netTypeArray[n].pixelClassData = pixelClassData[:, 2].copy()
                # Compute accuracy
                n_crop_types = len(netTypeArray[n].cropTypes)
                confusionMat = np.zeros((n_crop_types, n_crop_types))
                true_classes = np.array([pixelClassData[:, 1] == crop_type for crop_type in netTypeArray[n].cropTypes[:-1]])
                predicted_classes = pixelClassData[:, 2]
                for p in range(n_crop_types - 1):
                    for q in range(n_crop_types - 1):
                        is_class_q = predicted_classes[true_classes[p]] == netTypeArray[n].cropTypes[q]
                        confusionMat[p, q] = is_class_q.mean()
                        if p == q:
                            confusionMat[p, q] *= is_class_q.sum()
                confusionMat[:-1, -1] = 1 - confusionMat[:-1, :-1].sum(axis=1)
                trueI = np.logical_not(np.isin(pixelClassData[:, 1], netTypeArray[n].cropTypes[:-1])).squeeze()
                for q in range(n_crop_types - 1):
                    confusionMat[-1, q] = (predicted_classes[trueI] == netTypeArray[n].cropTypes[q]).mean()
                confusionMat[-1, -1] = 1 - confusionMat[-1, :-1].sum()
                confusionMat = confusionMat / confusionMat.sum()
                print("confusionMat:")
                print(confusionMat)
                print("accuracy: ", confusionMat.diagonal().sum())
                netTypeArray[n].results_field[:, :, i] = confusionMat.copy()
                netTypeArray[n].accuracy_field[0, i] = confusionMat.diagonal().sum()
                netTypeArray[n].outputIm = generate_classification_images(pixelClassData, rowI, colJ, maskImage.shape, 3)
                if showIm:
                    show_image(netTypeArray[n].outputIm.colorEst, "colorEst")
                if anyUpdate:
                    if showIm:
                        show_image(outputIm.colorTrue, "colorTrue")
                    atLeastOneStart = any(netType.isStart for netType in netTypeArray)
                    # Assign a default value to cropFraction
                    cropFraction = np.zeros((1, netTypeCount)) 
                    if atLeastOneStart:
                        pixelClassData[:, 2] = -1
                        for p in range(cropTypeCount):
                            pixelClassData[fixedCrop[p, :] == 1, 2] = p
                        for f in range(fieldCount):
                            alreadyFixed = any(pixelClassData[fieldStructArray[f].pixIndex[0], 2] == 1 for p in range(cropTypeCount))
                            if not alreadyFixed:
                                for n in range(netTypeCount):
                                    if netTypeArray[n].isStart:
                                        for p in range(len(netTypeArray[n].cropTypes) - 1):
                                            cropFraction[n] += np.sum(netTypeArray[n].pixelClassData[fieldStructArray[f].pixIndex] == netTypeArray[n].cropTypes[p]) / fieldStructArray[f].pixCount
                            if isFieldLevelOnly:
                                maxVal = np.amax(cropFraction)
                                netInd = np.argmax(cropFraction)
                                if maxVal > 0.5:
                                    if len(netTypeArray[netInd].cropTypes) == 2:
                                        pixelClassData[fieldStructArray[f].pixIndex, 2] = netTypeArray[netInd].cropTypes[0]
                                    else:
                                        cropfrac = [np.sum(netTypeArray[netInd].pixelClassData[fieldStructArray[f].pixIndex] == netTypeArray[netInd].cropTypes[p]) / fieldStructArray[f].pixCount for p in range(len(netTypeArray[netInd].cropTypes) - 1)]
                                        maxVal = np.amax(cropfrac)
                                        cropInd = np.argmax(cropfrac)
                                        pixelClassData[fieldStructArray[f].pixIndex, 2] = netTypeArray[netInd].cropTypes[cropInd]
                                else:
                                    pixelClassData[fieldStructArray[f].pixIndex, 2] = 4
                            else:
                                cropInd = sorted([i for i, val in enumerate(cropFraction) if val > 0], reverse=True)
                                if not cropInd:
                                    pixelClassData[fieldStructArray[f].pixIndex, 2] = cropTypeCount
                                else:
                                    for k in fieldStructArray[f].pixIndex.T.ravel():
                                        cropVal = [netTypeArray[n].pixelClassData[k] for n in range(netTypeCount) if netTypeArray[n].isStart and 0 <= netTypeArray[n].pixelClassData[k] <= cropTypeCount - 1]
                                        cropVal = [val for val in cropVal if val != 0]
                                        if not cropVal:
                                            pixelClassData[k, 2] = cropTypeCount
                                        else:
                                            if len(cropVal) == 1:
                                                pixelClassData[k, 2] = cropVal[0] - 1
                                            else:
                                                for j in cropInd:
                                                    if j in cropVal:
                                                        pixelClassData[k, 2] = j
                                                        break
            crop_len = len(cropList)
            confusionMat = np.zeros((crop_len, crop_len))
            correctNum = 0
            for p, _ in enumerate(cropList[:-1]):
                trueI = np.squeeze(np.where(pixelClassData[:, 1] == p))
                for q, _ in enumerate(cropList[:-1]):
                    confusionMat[p, q] = np.sum(pixelClassData[trueI, 2] == q) / len(trueI)
                    if p == q:
                        correctNum += confusionMat[p, q] * len(trueI)
                confusionMat[p, crop_len - 1] = 1 - np.sum(confusionMat[p, :crop_len - 1])
            trueI = np.squeeze(np.where(~np.isin(pixelClassData[:, 1], cropList[:crop_len - 1])))
            for q, _ in enumerate(cropList[:-1]):
                confusionMat[crop_len - 1, q] = np.sum(pixelClassData[trueI, 2] == q) / len(trueI)
            confusionMat[crop_len - 1, crop_len - 1] = 1 - np.sum(confusionMat[crop_len - 1, :crop_len - 1])
            correctNum += confusionMat[-1, -1] * len(trueI)
            accuracy = correctNum / len(imInd)
            print("confusionMat:")
            print(confusionMat)
            print("accuracy: ", accuracy)
            results_field[:, :, i] = confusionMat
            accuracy_field[0][i] = accuracy
            outputIm1 = generate_classification_images(pixelClassData, rowI, colJ, maskImage.shape, 3)
            if showIm:
                show_image(outputIm1.colorEst, "colorEst" )
            outputIm1.valEst = np.select([outputIm1.valEst == i for i in range(5)], [153, 147, 158, 146, 0], default=outputIm1.valEst)
            # write images
            cv2.imwrite(f'imOutput_largeImages_field/python_colorIm_{datestr}.png', outputIm1.colorEst / 255, params=(cv2.IMWRITE_PNG_COMPRESSION , 1))
            cv2.imwrite(f'imOutput_largeImages_field/python_cropType_{datestr}.png', outputIm1.valEst.astype(np.ubyte), params=(cv2.IMWRITE_PNG_COMPRESSION , 1))
    # combine results of different nets
    largeResults_field = {'evalDayArray': evalDayArray, 'results_field': results_field, 'accuracy_field': accuracy_field}
    scipy.io.savemat(f'AgricultureProcessing/bin/Release/net5.0-windows/PythonScripts/Classification/output/largeResults_field_{i + 1}.mat', largeResults_field)

def Classify():
    showIm = 0
    fieldMaskFile = r'\\arwen\HOME\Sina\LargeAreaPhenology\Crop_Inventory_reprojected_Sized.tif'
    maskImage = cv2.imread(fieldMaskFile)
    fieldID_truth_file = r'D:\abhijit\Crop_classification\growthEstimation\MB_large_fieldID_truth.mat'
    fieldID_truth_mat = scipy.io.loadmat(fieldID_truth_file)
    fieldIndex = np.squeeze(fieldID_truth_mat['fieldIndex'])
    trueValue = np.squeeze(fieldID_truth_mat['trueValue'])
    rowI = np.squeeze(fieldID_truth_mat['rowI'])
    colJ = np.squeeze(fieldID_truth_mat['colJ'])
    imInd = np.squeeze(fieldID_truth_mat['imInd'])
    pixCount = len(fieldIndex)
    isFieldLevelOnly = 1
    minMaturity = 0.3
    maxMaturity = np.array([0.8,0.8,0.8,0.75])
    startMaturity = 0.35
    minAvgMaurity = 0.45
    minFractionForDecision = 0.95

    netTypeList = [
        {'nanLocations': [6], 'cropTypes': [0,4]},
        {'nanLocations': [9,12], 'cropTypes': [1,2,4]},
        {'nanLocations': [15], 'cropTypes': [3,4]},
    ]

    fieldIndexSorted = np.sort(fieldIndex)
    indFieldIndexSorted = np.argsort(fieldIndex, kind='stable')
    dSortFieldIndex = np.diff(fieldIndexSorted)
    groupsSortFieldIndex = np.insert(dSortFieldIndex != 0, 0, True)
    invIndSortFieldIndex = np.argsort(indFieldIndexSorted, kind='stable')
    logIndFieldIndex = groupsSortFieldIndex[invIndSortFieldIndex]
    uniqueFieldIDs = fieldIndex[logIndFieldIndex]
    indUniqueFieldIDsSorted = np.argsort(uniqueFieldIDs, kind='stable')
    groupsSortFieldIndex = np.append(groupsSortFieldIndex, True)
    lengthGroupsSortFieldIndex = np.squeeze(np.diff(np.nonzero(groupsSortFieldIndex)))
    diffIndSortUniqueIDs = np.insert(np.diff(indUniqueFieldIDsSorted), 0, indUniqueFieldIDsSorted[0])
    indLengthGroupsSortFieldIndex = np.concatenate((np.array([1]), lengthGroupsSortFieldIndex)).cumsum()[:-1]
    indUniqueOrderedBySortFieldIndex = np.zeros(indLengthGroupsSortFieldIndex[-1])
    indLengthGroupsSortFieldIndex -= 1
    indUniqueOrderedBySortFieldIndex[indLengthGroupsSortFieldIndex] = diffIndSortUniqueIDs
    if (np.sum(lengthGroupsSortFieldIndex) != len(indUniqueOrderedBySortFieldIndex)):
        zerosToAdd = np.zeros(abs(np.sum(lengthGroupsSortFieldIndex) - len(indUniqueOrderedBySortFieldIndex)))
        indUniqueOrderedBySortFieldIndex = np.concatenate((indUniqueOrderedBySortFieldIndex, zerosToAdd))
    indUniqueOrderedBySortFieldIndex = np.cumsum(indUniqueOrderedBySortFieldIndex).astype(np.int64)
    IC = indUniqueOrderedBySortFieldIndex[invIndSortFieldIndex]

    fieldStructArray = []
    for i in range(0, 1):
        fieldStructArray.append(fieldStruct())
        fieldStructArray[i].ID = uniqueFieldIDs[i]
        fieldPixelIDs = np.squeeze(np.where(IC == i))
        fieldStructArray[i].pixIndex = fieldPixelIDs
        fieldStructArray[i].trueValue = trueValue[fieldPixelIDs[0]]
        fieldStructArray[i].pixCount = len(fieldPixelIDs)
    fieldCount = len(fieldStructArray)
    DataFolder = r'D:\abhijit\Crop_classification\growthEstimation\largeResults'
    DataList = np.array(os.listdir(DataFolder))
    sensorNames = np.array(['RCM','Sentinel 1','Sentinel 2'])
    cropTypeCount = 4
    sensorTypeCount = 3
    netTypeCount = 3
    fixedCrop = np.zeros((cropTypeCount,pixCount))
    evalDayArray = np.zeros((1,DataList.shape[1-1]))
    cropList = np.arange(0, cropTypeCount + 1)
    results_field = np.zeros((cropTypeCount + 1,cropTypeCount + 1,DataList.shape[1-1]))
    accuracy_field = np.zeros((1,DataList.shape[1-1]))
    netTypeArray = [initialize_net_type(item['nanLocations'], item['cropTypes'], pixCount, DataList) for item in netTypeList]
    pixelClassData = np.ones((pixCount,3)) * -1
    pixelClassData[:,0] = copy.copy(fieldIndex)
    trueValue_temp = copy.copy(trueValue)
    replacement_dict = {153: 0, 147: 1, 158: 2, 167: 2, 146: 3}
    for old_value, new_value in replacement_dict.items():
        trueValue_temp[trueValue_temp == old_value] = new_value
    trueValue_temp[trueValue_temp > 4] = 4
    pixelClassData[:,1] = copy.copy(trueValue_temp)
    outputIm = generate_classification_images(pixelClassData,rowI,colJ,maskImage.shape,2)
    currState = np.zeros((pixCount,cropTypeCount))
    currCovar = np.zeros((pixCount,cropTypeCount))
    currDateGDD = np.zeros((pixCount,2))
    accuLikeli = np.zeros((pixCount,cropTypeCount * netTypeCount))
    accuCount = np.zeros((pixCount,netTypeCount * sensorTypeCount))
    totalAccu = np.zeros((pixCount,netTypeCount))
    startAt = 0
    anyUpdate = 0
    evalDayArray = []

    # Determine the number of cores available
    num_cores = mp.cpu_count() // 2  # Use half of the cores or ot will crash the computer

    # Process first 40 files sequentially
    try:
        for i in range(startAt, startAt + 40):
            process(i, showIm, imInd, isFieldLevelOnly, minMaturity,
                    maxMaturity, startMaturity, minAvgMaurity,
                    minFractionForDecision, fieldCount, sensorNames,
                    fixedCrop, evalDayArray, cropList, results_field, accuracy_field,
                    netTypeArray, outputIm, currState, currCovar, currDateGDD, accuLikeli,
                    accuCount, totalAccu, DataList, DataFolder, netTypeCount, sensorTypeCount,
                    cropTypeCount, pixelClassData, fieldStructArray, rowI, colJ, maskImage,anyUpdate)
    except Exception as e:
        print("Exception occurred:", e)

    # Now, for the remaining files, create a list of delayed tasks
    tasks = (delayed(process)(i, showIm, imInd, isFieldLevelOnly, minMaturity,
                              maxMaturity, startMaturity, minAvgMaurity,
                              minFractionForDecision, fieldCount, sensorNames,
                              fixedCrop, evalDayArray, cropList, results_field, accuracy_field,
                              netTypeArray, outputIm, currState, currCovar, currDateGDD, accuLikeli,
                              accuCount, totalAccu, DataList, DataFolder, netTypeCount, sensorTypeCount,
                              cropTypeCount, pixelClassData, fieldStructArray, rowI, colJ, maskImage,anyUpdate) 
              for i in range(startAt + 40, len(DataList)))

    # Now, compute all tasks in parallel
    try:
        Parallel(n_jobs=num_cores)(tasks)
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc() 

    

if __name__ == '__main__':
    Classify()