import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio


class get_df_vs():

    def __init__(self):
        pass

    def plot_dist(self, df_data):
        fig = plt.figure(figsize=(16, 4))
        fig, axes = plt.subplots(nrows=1, ncols=3)
        df_data['label'].value_counts().plot(kind='bar', ax=axes[0], title='label dist')
        df_data['count'].value_counts().plot(kind='bar', ax=axes[1], title='count persons')
        df_data['countFrames'].plot(kind='hist', ax=axes[2], title='count frames')
        plt.show()
        tmp = df_data['label'].value_counts()
        nr_false, nr_true = tmp[0], tmp[1]

        print('Nr_True: ', nr_true)
        print('Nr_false: ', nr_false)

    def plot_label_dist(self, df_data):
        fig = plt.figure(figsize=(6, 4))

        df_data['label'].value_counts().plot(kind='bar',
                                             color='r')
        print(df_data['label'].value_counts())
        plt.ylabel('Nr samples')
        plt.savefig('./plots/introduction/distribution_data.png')

    def plot_count_frames(self, df_data):
        fig = plt.figure(figsize=(6, 4))

        df_data = df_data[df_data['countFrames'] < 500]

        plt.hist(df_data['countFrames'], color='r', bins=100)

        plt.xlabel('Number of frames')
        plt.ylabel('Nr samples')
        plt.savefig('./plots/introduction/count_frames.png')
        plt.show()

    def plot_segmentation(self, df_data):
        fig = plt.figure(figsize=(6, 4))
        df_true = df_data[df_data['label'] == True]
        df_true['segmentation'].value_counts().plot(kind='bar', color='r')
        plt.ylabel('Nr samples')
        plt.savefig('./plots/introduction/segmentation.png')
        plt.show()


class bgs_vs():

    def __init__(self):
        pass

    ######## public functions ####################
    def play_videos(self,df_data):

        df_true = df_data[df_data['label'] == True]
        df_false = df_data[df_data['label'] == False]

        i_true  = 0
        i_false = 0
        j_true  = 0
        j_false = 0

        len_true = len(df_true)
        len_false = len(df_false)
        images = []
        while (1):
            if (i_true >= len_true):
                i_true = 0
            if (i_false >= len_false):
                i_false = 0
            if (j_true >= len(df_true.iloc[i_true]['frames'])):
                j_true  = 0
                i_true += 1
            if (j_false >= len(df_false.iloc[i_false]['frames'])):
                j_false = 0
                i_false += 1

            path_true = 'data/raw/configured_raw/' + df_true.iloc[i_true]['name'] + '/visualized/' + \
                        df_true.iloc[i_true]['frames'][j_true]
            path_false = 'data/raw/configured_raw/' + df_false.iloc[i_false]['name'] + '/visualized/' + \
                         df_false.iloc[i_false]['frames'][j_false]
            path_true_BGS = 'data/interim/BGS/bit_8/' + df_true.iloc[i_true]['frames'][j_true]
            path_false_BGS = 'data/interim/BGS/bit_8/' + df_false.iloc[i_false]['frames'][j_false]

            frame_true     = cv2.imread(path_true, 0)
            frame_false    = cv2.imread(path_false, 0)
            frame_true_BGS = cv2.imread(path_true_BGS, -1)
            frame_false_BGS = cv2.imread(path_false_BGS, -1)

            frame = np.concatenate((frame_false, frame_true), axis=0)
            # frame_BGS = np.concatenate((frame_false_BGS, frame_true_BGS), axis=1)
            # frame = np.concatenate((frame, frame_BGS), axis=0)
            images.append(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Image False Id: ' + str(i_false), (20, 20), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, 'Image True Id: ' + str(i_true), (20, 260), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            time.sleep(0.03)

            j_true += 2
            j_false += 2

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                i_false += 1
            if key == ord('a'):
                if i_false == 0:
                    i_false = len_false - 1
                else:
                    i_false -= 1
            if key == ord('x'):
                i_true += 1
            if key == ord('z'):
                if i_true == 0:
                    i_true = len_true - 1
                else:
                    i_true -= 1

        cv2.destroyAllWindows()
        imageio.mimsave('./plots/gif/data.gif', images)

    def show_process(self, name, index):

        BGI_16 = self.load_BGI_raw(name)
        BGI_8 = self.load_BGI_vis(name)
        BGI_8 = cv2.cvtColor(BGI_8, cv2.COLOR_BGR2GRAY)
        ratio = np.max(BGI_16) / np.max(BGI_8)
        BGI_16_8 = (BGI_16 / ratio).astype('uint8')

        image_16, name_16 = self.load_IMG_raw(name, index)
        image_8, name_8 = self.load_IMG_vis(name, index)
        image_8 = cv2.cvtColor(image_8, cv2.COLOR_BGR2GRAY)
        image_16_8 = (image_16 / ratio).astype('uint8')

        mask = self.calc_mask(BGI_16, image_16)
        mask_8 = (mask * 255.).astype('uint8')

        img_BGI_16 = np.multiply(image_16, mask)
        img_BGI_8 = np.multiply(image_8, mask)
        img_BGI_16_8 = (img_BGI_16 / ratio).astype('uint8')

        img1 = np.concatenate((BGI_16_8, image_16_8, mask_8, img_BGI_16_8))
        img2 = np.concatenate((BGI_8, image_8, mask_8, img_BGI_8))
        img = np.concatenate((img1, img2), axis=1)

        self.show_image(img)

    def show_reconstructed_pic(self, df, index_m, index_p):

        for i in range(4):

            name = df.iloc[index_m + i]['name']

            BGI_16 = self.load_BGI_raw(name)
            BGI_8 = self.load_BGI_vis(name)
            ratio = np.max(BGI_16) / np.max(BGI_8)
            BGI_16_8 = (BGI_16 / ratio).astype('uint8')

            path = 'data/interim/BGS/bit_16/' + df.iloc[index_m + i]['frames'][index_p]
            image_BGS_16 = cv2.imread(path, -1)

            image_BGS_16_8 = (image_BGS_16 / ratio).astype('uint8')

            path = 'data/raw/configured_raw/' + name + '/raw/' + df.iloc[index_m + i]['frames'][index_p]
            image_16 = cv2.imread(path, -1)
            image_16_8 = (image_16 / ratio).astype('uint8')


            if (i == 0):
                img = np.concatenate((BGI_16_8, image_16_8, image_BGS_16_8))
            else:
                img1 = np.concatenate((BGI_16_8, image_16_8, image_BGS_16_8))
                img = np.concatenate((img, img1), axis=1)

        self.show_image(img)

    def load_BGI_raw(self, name):
        path = 'data/raw/configured_raw/' + name + '/depth-references/'
        pic = os.listdir(path)[1]
        path = path + pic
        BGI = cv2.imread(path, -1)
        return BGI

    ####### private functions #####################
    def load_BGI_vis(self, name):
        path = 'data/raw/configured_raw/' + name + '/depth-references/visualized/'
        pic = os.listdir(path)[0]
        path = path + pic
        BGI = cv2.imread(path, -1)
        return BGI

    def load_IMG_raw(self, name, index):
        path = 'data/raw/configured_raw/' + name + '/raw/'
        name = os.listdir(path)[index]
        path = path + name
        image = cv2.imread(path, -1)
        image = image
        return image, name

    def load_IMG_vis(self, name, index):
        path = 'data/raw/configured_raw/' + name + '/visualized/'
        name = os.listdir(path)[index]
        path = path + name
        image = cv2.imread(path, -1)
        image = image
        return image, name

    def calc_mask(self, BGI, image):
        img_BGS = cv2.subtract(BGI, image)
        img_mask = np.zeros(img_BGS.shape).astype('uint8')
        img_mask[img_BGS > 100] = 255
        ret, thresh = cv2.threshold(img_mask, 100, 260, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_image = np.zeros((img_mask.shape))

        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > 100):
                cv2.drawContours(new_image, [contour], -1, (255, 255, 255), thickness=-1)

        return (new_image / 255.).astype('uint8')

    def show_image(self, image):
        while True:
            cv2.imshow('BG', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()


class path_gen_vs():

    def __init__(self,df,dict_pg):
        self.df = df
        self.dict_pd = dict_pg


    def play_videos(self, heigth):
        i = 65
        j = 0
        len_df = len(self.df)
        images = []
        while (1):
            if (i >= len_df):
                i = 0
            if (j >= len(self.df.iloc[i]['frames'])):
                j = 0
                heigth += 30

            # Load in image
            name = self.df.iloc[ i]['name']

            path = 'data/raw/configured_raw/' + name + '/depth-references/visualized/'
            pic = os.listdir(path)[0]
            path = path + pic
            BGI = cv2.imread(path, -1)
            BGI = cv2.cvtColor(BGI, cv2.COLOR_BGR2GRAY)

            path = 'data/interim/BGS/bit_8/' + self.df.iloc[i]['frames'][j]
            frame = cv2.imread(path, -1)
            frame_proc, data = self.slice_frame(frame, heigth)

            if (j == 0):
                frame_path = np.zeros(frame_proc.shape).astype('uint8')

            # draw on image and concatenate
            self.draw_circle(frame_path, data)

            frame_1 = np.concatenate((BGI,frame), axis=0)
            frame_2 = np.concatenate((frame_proc, frame_path),axis=0)

            frame_con = np.concatenate((frame_1,frame_2), axis = 1)
            self.write_text(frame_con, i, heigth)
            images.append(frame_con)
            # show image
            cv2.imshow('frame', frame_con)
            time.sleep(0.03)

            # Controls GUI
            j += 2
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                i += 1
            if key == ord('a'):
                if i == 0:
                    i = len_df - 1
                else:
                    i -= 1

            if key == ord('z'):
                heigth += 1
            if key == ord('x'):
                heigth -= 1

        cv2.destroyAllWindows()
        imageio.mimsave('./plots/gif/PD.gif', images)


    def draw_circle(self, frame_path, data):

        for i in range(0, len(data), 4):
            if (data[i + 3] != 0):
                cv2.circle(frame_path, (int(data[i]), int(data[i + 1])), 1, (255, 255, 255), -1)

    def write_text(self, frame, i, heigth):

        cv2.putText(frame,
                    'Image  Id: ' + str(i) + ' and heigth: ' + str(heigth),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

    def slice_frame(self, frame, heigth):

        frame_mask                 = np.zeros(frame.shape).astype('uint8')
        frame_mask[frame < heigth] = 255
        frame_mask[frame == 0]     = 0

        _, contours, _             = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sliced_frame               = np.zeros((frame_mask.shape)).astype('uint8')

        data_np                    = np.zeros(self.dict_pd['max_p_h'] * 4)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > self.dict_pd['area']):
                cv2.drawContours(sliced_frame, [contour], -1, (255, 255, 255), thickness=-1)
                M  = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the contour and center of the shape on the image
                cv2.circle(sliced_frame, (cX, cY), 7, (0, 0, 0), -1)

                if (i < 4):
                    data_np[4 * i:4 * i + 4] = np.array([cX, cY, heigth, area])

        return sliced_frame, data_np


class Error_analysis_LSTM():

    def __init__(self):
        pass

    def plot_val_distr_ROC(self,path,bins):

        windows = os.listdir(path)
        windows = self.sort_windows(windows)

        for window in windows:

            dict_ = {}
            dict_['AUC'] = []
            max_ = 0

            path_w = path + window + '/'
            pickles = os.listdir(path_w)

            if(len(pickles)>0):
                for pickle_ in pickles:

                    path_p = path_w + pickle_
                    tmp = hf.pickle_load(path_p)
                    AUC = tmp['AUC']

                    dict_['AUC'].append(AUC)
                    if (AUC > max_):
                        history = tmp['hist']
                        TN      = tmp['TN']
                        FN      = tmp['FN']
                        max_    = AUC


                fig = plt.figure(figsize=(16, 4))

                ax1 = plt.subplot(131)
                ax1.hist(dict_['AUC'],bins= bins)
                plt.title('AUC Distr ' + window)
                plt.xticks(rotation=70)

                ax2 = plt.subplot(132)
                ax2.plot(history['val_loss'], color='g')
                ax2.plot(history['loss'], color='r')
                plt.title('Validation curve')
                plt.xticks(rotation=70)

                ax3 = plt.subplot(133)
                ax3.plot(FN, TN)
                plt.title('ROC with AUC: '+str(max_))
                plt.xticks(rotation=70)

                plt.show()

    def distr_error(self,path, bins_f=5,bins_t=5, mode='max'):

        windows = os.listdir(path)
        windows = self.sort_windows(windows)
        ev_true = []
        ev_false = []
        AUC = []

        for window in windows:

            dict_ = {}
            dict_['AUC'] = []
            max_ = 0

            path_w = path + window + '/'
            pickles = os.listdir(path_w)
            if(len(pickles)>0):
                for pickle_ in pickles:

                    path_p = path_w + pickle_
                    tmp = hf.pickle_load(path_p)

                    AUC_ = tmp['AUC']

                    if (AUC_ > max_):

                        if (mode == 'max'):
                            ev_true_ = self.get_max(tmp['ev_true'])
                            ev_false_ = self.get_max(tmp['ev_false'])
                        if (mode == 'ravel'):
                            ev_true_ =  self.ravel_array(tmp['ev_true'])
                            ev_false_ = self.ravel_array(tmp['ev_false'])

                        max_ = AUC_

                ev_true.append(ev_true_)
                ev_false.append(ev_false_)
                AUC.append(max_)

        if(len(ev_true)>0):
            self.plot_hist_per_3(ev_true, ev_false, AUC, windows,  bins_f,bins_f)

    def plot_hist_per_3(self,ev_true, ev_false, AUC, windows,  bins_f=5,bins_t=5):

        iterations = math.ceil(len(ev_true) / 3)
        modulus = len(ev_true) % 3

        for i in range(iterations):
            if (i != iterations - 1 or modulus == 0):

                fig = plt.figure(figsize=(16, 4))

                ax1 = plt.subplot(131)
                ax1.hist(ev_true[3 * i],  bins=bins_t, color='g', alpha=0.5)
                ax1.hist(ev_false[3 * i], bins=bins_f, color='r', alpha=0.5)
                plt.title(windows[3 * i] + '        AUC: ' + str(AUC[3 * i]))
                plt.xticks(rotation=70)

                ax1 = plt.subplot(132)
                ax1.hist(ev_true[3 * i + 1],  bins=bins_t, color='g', alpha=0.5)
                ax1.hist(ev_false[3 * i + 1], bins=bins_f, color='r', alpha=0.5)
                plt.title(windows[3 * i + 1] + '        AUC: ' + str(AUC[3 * i + 1]))
                plt.xticks(rotation=70)

                ax1 = plt.subplot(133)
                ax1.hist(ev_true[3 * i + 2],  bins=bins_t, color='g', alpha=0.5)
                ax1.hist(ev_false[3 * i + 2], bins=bins_f, color='r', alpha=0.5)
                plt.title(windows[3 * i + 2] + '        AUC: ' + str(AUC[3 * i + 2]))
                plt.xticks(rotation=70)

                plt.show()

            else:
                if (modulus == 2):

                    fig = plt.figure(figsize=(16, 4))

                    ax1 = plt.subplot(121)
                    ax1.hist(ev_true[3 * i], bins=bins_t, color='g', alpha=0.5)
                    ax1.hist(ev_false[3 * i], bins=bins_f, color='r', alpha=0.5)
                    plt.title(windows[3 * i] + '        AUC: ' + str(AUC[3 * i]))
                    plt.xticks(rotation=70)

                    ax1 = plt.subplot(122)
                    ax1.hist(ev_true[3 * i + 1], bins=bins_t, color='g', alpha=0.5)
                    ax1.hist(ev_false[3 * i + 1], bins=bins_f, color='r', alpha=0.5)
                    plt.title(windows[3 * i + 1] + '        AUC: ' + str(AUC[3 * i + 1]))
                    plt.xticks(rotation=70)

                    plt.show()


                else:
                    fig = plt.figure(figsize=(16, 4))
                    plt.hist(ev_true[3 * i], bins=bins_t, color='g', alpha=0.5)
                    plt.hist(ev_false[3 * i], bins=bins_f, color='r', alpha=0.5)
                    plt.title(windows[3 * i] + '        AUC: ' + str(AUC[3 * i]))
                    plt.xticks(rotation=70)
                    plt.plot()

    def sort_windows(self,windows):
        num = []
        for window in windows:
            num.append(int(window[2:]))
            num = sorted(num)
        windows_fin = []

        for num_ in num:
            for window in windows:
                if (str(num_) == window[2:]):
                    windows_fin.append(window)
        return windows_fin

    def get_max(self,array):
        max_array = []
        for x in array:
            max_array.append(max(x))

        return max_array

    def ravel_array(self,array):
        ravel_array = []
        for x in array:
            ravel_array.extend(x)

        return ravel_array


class video_error_display():

    def __init__(self,dict_config):
        self.path = dict_config['path']
        self.df   = dict_config['df']
        self.eval = self.get_best_models(self.path)

        self.eval = self.conf_eval(self.eval,self.df)


    def conf_eval(self,eval_d,df):
        i_true  = 0
        i_false = 0

        ev_a_true  = eval_d['ev_true']
        ev_a_false = eval_d['ev_false']

        ev_a      = []

        for i in range(len(df)):

            len_frames = len(df['frames'].iloc[i])
            boolean = df['label'].iloc[i]

            if(boolean == True):

                eval = ev_a_true[i_true]
                i_true   += 1

            elif(boolean == False):
                eval = ev_a_false[i_false]
                i_false   += 1

            print(len_frames)
            print(len(eval))
            difference = len_frames-len(eval)
            zeros      = np.zeros(difference)

            print(zeros)
            eval       = np.concatenate(zeros,eval)

            print(eval)
            ev_a.append(eval)

        return ev_a

        return


    def error_visualization(self):



        i = 0
        j = 0
        len_df = len(self.df)
        while (1):
            if (i >= len_df):
                i = 0
            if (j >= len(self.df.iloc[i]['frames'])):
                j = 0



            # Load in image
            path = 'data/interim/BGS/bit_8/' + self.df.iloc[i]['frames'][j]
            frame = cv2.imread(path, -1)





            # show image
            cv2.imshow('frame', frame)
            time.sleep(0.03)

            # Controls GUI
            j += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                i += 1
            if key == ord('a'):
                if i == 0:
                    i = len_df - 1
                else:
                    i -= 1


    def get_best_models(self,path):

        windows = os.listdir(path)

        for window in windows:

            dict_ = {}
            dict_['ev_false'] = []
            dict_['ev_true'] = []

            max_ = 0

            path_w = path + window + '/'
            pickles = os.listdir(path_w)

            for pickle_ in pickles:


                path_p = path_w + pickle_
                tmp = hf.pickle_load(path_p)
                AUC = tmp['AUC']



                if (AUC > max_):
                    ev_true_  = tmp['ev_true']
                    ev_false_ = tmp['ev_false']

                    max_ = AUC


                dict_['ev_false'].append(ev_false_)
                dict_['ev_true'].append(ev_true_)

        return dict_






