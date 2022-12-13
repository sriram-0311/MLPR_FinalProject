import glob
from tqdm import tqdm
import cv2
import mediapipe as mp
import pandas as pd


if __name__ == '__main__':
    #folder_path = '/home/trudes/EECE5644/MLPR_FinalProject/data/asl_alphabet_train/asl_alphabet_train/'
    folder_path = '/home/trudes/EECE5644/MLPR_FinalProject/data/diversity_test/'
    #folder_path = "/home/trudes/EECE5644/MLPR_FinalProject/data/asl_alphabet_test/asl_alphabet_test/"
    #pandas_output_filename = "preprocessed_asl_train_reformatted.csv"
    pandas_output_filename = "preprocessed_diversity_test_reformatted.csv"
    
    bin_json_list = glob.glob(folder_path+'/*/*')
    #bin_json_list = glob.glob(folder_path+'*')
    print(bin_json_list)

    #hand detection model
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    

    #convert each image to model
    processed_data = []
    # pandas.DataFrame(data, index, columns)
    for filepath in tqdm(bin_json_list):
        mat_filename = filepath.split('/')[-1]
        #print(mat_filename)
        if mat_filename.endswith('.png'):
            #processed_filename = mat_filename.strip('.jpg')
            
            label = filepath.split('/')[-2]
            #label = filepath.split('/')[-1].strip('_test.jpg')
            print(label)
            img = cv2.imread(filepath)
            #print(img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            #print(results)

            if results.multi_hand_landmarks:
                data = {}
                for handLms in results.multi_hand_landmarks:
                    #print(handLms)
                    wrist_loc = handLms.landmark[0]
                    # test = handLms.landmark[0].x - handLms.landmark[1].x
                    # print(test)
                    # proccessed_datapoint = handLms.landmark
                    # print(type(test))
                    # proccessed_datapoint.append(label)
                    # processed_data.append(proccessed_datapoint)
                    #print(mpHands.HandLandmark(0).name)
                    proccessed_datapoint = []
                    for id, lm in enumerate(handLms.landmark):
                        x = lm.x - wrist_loc.x
                        y = lm.y - wrist_loc.y
                        z = lm.z - wrist_loc.z
                        proccessed_datapoint.append(x)
                        proccessed_datapoint.append(y)
                        proccessed_datapoint.append(z)
                proccessed_datapoint.append(label)
                processed_data.append(proccessed_datapoint)
                #print(proccessed_datapoint)
                        # #print(mpHands.HandLandmark(id).name)
                        # data[mpHands.HandLandmark(id).name] = lm
                        # #print(label)
                        # proccessed_datapoint = [data, label]
                        # #print(proccessed_datapoint)
                        # processed_data.append(proccessed_datapoint)
                        # #bin_filename = mat_filename+'.bin'
                        
                    #     # with open(os.path.join(sigmf_path,bin_filename), 'wb') as handle:
                    #     #     handle.write(binary_format)
    column_names = ["WRISTX", "WRISTY", "WRISTZ", 
    "THUMB_CMCX", "THUMB_CMCY", "THUMB_CMCZ",
    "THUMB_MCPX", "THUMB_MCPY", "THUMB_MCPZ", 
    "THUMB_IPX", "THUMB_IPY", "THUMB_IPZ", 
    "THUMB_TIPX", "THUMB_TIPY", "THUMB_TIPZ", 
    "INDEX_FINGER_MCPX","INDEX_FINGER_MCPY","INDEX_FINGER_MCPZ",
    "INDEX_FINGER_PIPX","INDEX_FINGER_PIPY","INDEX_FINGER_PIPZ",
    "INDEX_FINGER_DIPX","INDEX_FINGER_DIPY","INDEX_FINGER_DIPZ",
    "INDEX_FINGER_TIPX","INDEX_FINGER_TIPY","INDEX_FINGER_TIPZ",
    "MIDDLE_FINGER_MCPX","MIDDLE_FINGER_MCPY","MIDDLE_FINGER_MCPZ",
    "MIDDLE_FINGER_PIPX","MIDDLE_FINGER_PIPY","MIDDLE_FINGER_PIPZ",
    "MIDDLE_FINGER_DIPX","MIDDLE_FINGER_DIPY","MIDDLE_FINGER_DIPZ",
    "MIDDLE_FINGER_TIPX","MIDDLE_FINGER_TIPY","MIDDLE_FINGER_TIPZ",
    "RING_FINGER_MCPX","RING_FINGER_MCPY","RING_FINGER_MCPZ",
    "RING_FINGER_PIPX","RING_FINGER_PIPY","RING_FINGER_PIPZ",
    "RING_FINGER_DIPX","RING_FINGER_DIPY","RING_FINGER_DIPZ",
    "RING_FINGER_TIPX","RING_FINGER_TIPY","RING_FINGER_TIPZ", 
    "PINKY_MCPX","PINKY_MCPY","PINKY_MCPZ",
    "PINKY_PIPX","PINKY_PIPY","PINKY_PIPZ",
    "PINKY_DIPX","PINKY_DIPY","PINKY_DIPZ",
    "PINKY_TIPX","PINKY_TIPY","PINKY_TIPZ", "LABEL"]
    df = pd.DataFrame(processed_data, columns=column_names)  
    print(df.head())    
    df.to_csv(pandas_output_filename, encoding='utf-8', index=False)

    # #second format
    # columns_names = processed_data[0][0].keys()
    # columns_names.append('label')
    # df2 = pd.DataFrame([], columns=columns_names)
    # for row in processed_data:
    #     feature_dict_as_lst = []
    #     features_dict = row[0]
    #     label = row[1]
    #     for key in features_dict.keys():
    #         feature_dict_as_lst.append(features_dict[key])
    #     updated_row = pd.DataFrame(feature_dict_as_lst, columns=columns_names)
    #     df2.append(updated_row)
    # print(df2.head())



    # # columns = processed_data.keys()
    # # reformatted_processed_data = []
    # # for item in processed_data:
    # #     features_dict = processed_data[0]
    # #     row = 
    # #     label = processed_data[1]
    # #     reformatted_processed_data.append()
    # # columns.append('label')
    