import os
import shutil

import cv2
from tqdm import tqdm

from src.dst.helper.apply_mp import *
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt




class get_df():

    def __init__(self,path):
        self.path   = path

    def get_df_data(self,list_names):
        list_data = []

        array_f   = []

        count_unlabeled_data = 0
        count_labeled_data = 0
        count_movie = 0

        for name in list_names:
            path = self.path + name + '/passages.json'
            passages = pd.read_json(path)
            for i in range(len(passages['passages'])):
                try:
                    dict_ = {'label': passages['passages'][i]['label']['anomaly'],
                             'count': passages['passages'][i]['label']['personCountDelta'],
                             'frames': passages['passages'][i]['frames'],
                             'name': name,
                             'movieID': count_movie}
                    count_labeled_data += 1
                    count_movie += 1

                except Exception as e:

                    dict_ = {'label': 'und',
                             'count': 'und',
                             'frames': passages['passages'][i]['frames'],
                             'name': name,
                             'movieID': count_movie}

                    count_movie += 1
                    count_unlabeled_data += 1

                if(dict_['frames'] not in array_f):
                    array_f.append(dict_['frames'])
                    list_data.append(dict_)

        df_data = pd.DataFrame(list_data, columns=['label', 'count', 'frames', 'name', 'movieID'])

        df_data = df_data.apply(self.apply_countFrames, axis=1)
        df_data = df_data[df_data['countFrames']>5]

        return df_data

    def apply_countFrames(self,row):
        row['countFrames'] = len(row['frames'])
        return row

class Move_p():

    def __init__(self):
        self.path = './data/raw/configured_raw/'

    def move_pictures(self,df):
        count = 0
        array = []
        for i in tqdm(range(len(df))):
            for j in range(len(df.iloc[i]['frames'])):
                try:
                    ### bit 16 #######
                    src =  self.path+ df.iloc[i]['name'] + '/raw/' + df.iloc[i]['frames'][j]
                    dst = 'data/interim/no_PP/bit_16/' + df.iloc[i]['frames'][j]
                    shutil.copy2(src, dst)

                    ### bit 8  #######
                    src =  self.path + df.iloc[i]['name'] + '/visualized/' + df.iloc[i]['frames'][j]
                    dst = 'data/interim/no_PP/bit_8/' + df.iloc[i]['frames'][j]
                    shutil.copy2(src, dst)
                except:
                    count += 1
                    if(i not in array):
                        array.append(i)
        for i in array:
            df.drop(df.index[i], inplace=True)
        return df


class BGS():

    def __init__(self,dict_c):
        self.path   = './data/raw/configured_raw/'
        self.dict_c = dict_c


    #### PUBLIC ########################################
    ####################################################

    def main(self ,df_data):


        for i in tqdm(range(len(df_data))):
            name    = df_data['name'].iloc[i]
            BGI     = self._load_BGI_raw(name)
            BGI_16  = self._load_BGI_raw(name)
            BGI_8   = self._load_BGI_vis(name)

            BGI_8   = cv2.cvtColor(BGI_8, cv2.COLOR_BGR2GRAY)
            ratio   = np.max(BGI_16 ) /np.max(BGI_8)

            for j in range(len(df_data['frames'].iloc[i])):
                src      = self.path + name + '/raw/' + df_data.iloc[i]['frames'][j]
                image    = cv2.imread(src, -1)
                mask     = self._calc_mask(BGI, image)
                img_BGI  = np.multiply(image, mask)

                path = 'data/interim/BGS/bit_16/' + df_data.iloc[i]['frames'][j]
                cv2.imwrite(path, img_BGI)

                img_BGI_8 = (img_BGI / ratio).astype('uint8')
                path = 'data/interim/BGS/bit_8/' + df_data.iloc[i]['frames'][j]
                cv2.imwrite(path, img_BGI_8)
    ####Private ############################################
    ########################################################

    def _load_BGI_raw(self, name):
        path    = self.path + name + '/depth-references/'

        pics     = os.listdir(path)

        for pic in pics:
            if('depth' in pic):
                string = pic
        path    = path + string
        BGI     = cv2.imread(path, -1)
        return BGI

    def _load_BGI_vis(self, name):
        path    = self.path + name + '/depth-references/visualized/'
        pic     = os.listdir(path)[0]
        path    = path + pic
        BGI     = cv2.imread(path, -1)
        return BGI

    def _load_IMG_raw(self, name, index):
        path    = self.path + name + '/raw/'
        name    = os.listdir(path)[index]
        path    = path + name
        image   = cv2.imread(path, -1)
        image   = image
        return image, name

    def _load_IMG_vis(self, name, index):
        path    = self.path + name + '/visualized/'
        name    = os.listdir(path)[index]
        path    = path + name
        image   = cv2.imread(path, -1)
        image   = image
        return image, name

    def _calc_mask(self, BGI, image):

        threshold                     = self.dict_c['threshold']
        min_area                      = self.dict_c['area']


        img_BGS                       = cv2.subtract(BGI, image)
        img_mask                      = np.zeros(img_BGS.shape).astype('uint8')
        img_mask[img_BGS > threshold] = 255
        ret, thresh                   = cv2.threshold(img_mask, 100, 260, cv2.THRESH_BINARY)
        im2, contours, hierarchy      = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_image                     = np.zeros((img_mask.shape))

        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > min_area):
                cv2.drawContours(new_image, [contour], -1, (255, 255, 255), thickness=-1)

        return (new_image / 255.).astype('uint8')

class get_min_max_h():

    def __init__(self,path_i,path_o):
        self.path_i = path_i
        self.path_o = path_o

        self.max  = -1
        self.min  = 10000

    def main(self):

        pics = os.listdir(self.path_i)
        for i in tqdm(range(len(pics))):

            path_v = self.path_i+pics[i]

            img = cv2.imread(path_v,-1)

            max_ = np.max(img)
            min_ = np.min(img)

            if(max_ > self.max):
                self.max = max_
            if(min_ < self.min):
                self.min = min_

        dict_ = {
            'min':self.min,
            'max':self.max
        }
        df = pd.Series(dict_).to_json(self.path_o)

class path_Generation():
    def __init__(self, df, dict_c):
        self.df = df
        self.dict_c = dict_c

        self.nr_features = dict_c['nr_features']
        self.nr_contours = dict_c['nr_contours']
        self.area        = dict_c['area']


        self.sum_        = 1000

        self.resolution   = np.arange(self.dict_c['min_h'], self.dict_c['max_h'], self.dict_c['resolution'])
        self.track_points = tracker(dict_c)


    ######## Main function #############################
    ######################################################


    def main(self,*args):
        self.df =  apply_by_multiprocessing(self.df, self._get_points_movie, axis=1, workers=8)
        # self.df = self.df.apply(self._get_points_movie, axis=1)
        return self.df

    def configure_frame(self,df,i,j,heigth):

        # Load in image
        path = 'data/interim/BGS/bit_8/' + df.iloc[i]['frames'][j]
        frame = cv2.imread(path, -1)
        data_normal,frame_proc = self._slice_frame(frame, heigth,j)
        data_pos, data_v       = self.track_points.main_tracker(j, heigth, data_normal)

        if (j == 0):
            self.frame_path = np.zeros(frame_proc.shape).astype('uint8')

        # draw on image and concatenate
        self._draw_circle(self.frame_path, data_normal)
        frame_con = np.concatenate((frame, frame_proc, self.frame_path), axis=1)
        self._write_text(frame_con, i,j,heigth)
        self._write_data(frame_con,data_pos)

        frame_con = cv2.cvtColor(frame_con, cv2.COLOR_GRAY2RGB)

        return frame_con

    def _get_points_movie(self, row):

        data_movie_p  = np.zeros((len(row['frames']), len(self.resolution) * self.nr_contours* self.nr_features))
        data_movie_v  = np.zeros((len(row['frames']), len(self.resolution) * self.nr_contours *( self.nr_features-1)))



        for j, frame in enumerate(row['frames']):

            path           = 'data/interim/BGS/bit_8/' + frame
            image          = cv2.imread(path, -1)
            frame_pt       = np.zeros(len(self.resolution) * self.nr_contours * self.nr_features)
            frame_vt       = np.zeros(len(self.resolution) * self.nr_contours * (self.nr_features-1))

            for i, heigth in enumerate(self.resolution):

                data,sliced_frame = self._slice_frame(image, heigth,j)
                data_pos,data_v   = self.track_points.main_tracker(j,heigth,data)


                if (len(data) != 0):
                    frame_pt[self.nr_features * self.nr_contours* i:self.nr_features * self.nr_contours * i + self.nr_features * self.nr_contours] = data_pos
                    frame_vt[(self.nr_features-1) * self.nr_contours* i:(self.nr_features-1) * self.nr_contours * i + (self.nr_features-1)* self.nr_contours] = data_v


            data_movie_p[j, :] = frame_pt
            data_movie_v[j, :] = frame_vt

        row['data_p']    = data_movie_p
        row['data_v']      = data_movie_v

        return row

    def _slice_frame(self, frame, heigth,frame_index):


        frame_mask                 = np.zeros(frame.shape).astype('uint8')
        frame_mask[frame < heigth] = 255
        frame_mask[frame == 0]     = 0


        _, contours, _             = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sliced_frame               = np.zeros((frame_mask.shape)).astype('uint8')

        data_np                    = np.zeros(self.nr_features * self.nr_contours)
        count = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > self.area):
                cv2.drawContours(sliced_frame, [contour], -1, (255, 255, 255), thickness=-1)
                M  = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.circle(sliced_frame, (cX, cY), 7, (0, 0, 0), -1)

                # cX = cX*(area-self.area)
                # cY = cY*(area-self.area)

                if (count < self.nr_contours):
                    data_np[self.nr_features * count:self.nr_features * count + self.nr_features] = np.array([cX, cY, area])
                    count += 1

        return data_np,sliced_frame

    def _draw_circle(self, frame_path, data):
        for i in range(0, len(data), self.nr_features):
            if (data[i + 2] != 0):
                cv2.circle(frame_path, (int(data[i]), int(data[i + 1])), 1, (255, 255, 255), -1)

    def _write_data(self, frame, data):
        for i in range(0,len(data),self.nr_features):

            string = 'cX: '+str(data[i])+', cY: '+str(data[i+1])+', Area: '+str(data[i+2])

            cv2.putText(frame,
                        string,
                        (20, 10*i+100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255), 1,
                        cv2.LINE_AA)

    def _write_text(self, frame, i, j,heigth):

        cv2.putText(frame,
                    'Image  Id: ' + str(i) + ' and heigth: ' + str(heigth)+'. Label =  '+str(self.df.iloc[i]['label'])+
                    '. Framenumber : '+str(j)+'/'+str(len(self.df.iloc[i]['frames'])),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

class tracker():

    def __init__(self,dict_c):
        self.dict_c        = dict_c
        self.nr_contours   = self.dict_c['nr_contours']
        self.nr_features   = self.dict_c['nr_features']


        self.resolution  = np.arange(self.dict_c['min_h'], self.dict_c['max_h'], self.dict_c['resolution'])



        self.previous_f  = {}
        self.state       = {}
        self.track_frame = {}
        self.change      = {}
        self.change_copy = ""
        self._config_prev()

    def main_tracker(self,frame_index,heigth,data_pos):
        if (frame_index == 0):
            tmp = []
            for i in range(0, self.nr_contours * self.nr_features, self.nr_features):
                tmp.append([data_pos[i], data_pos[i + 1]])
            if (frame_index != self.track_frame[heigth]):
                self.previous_f[heigth] = tmp

            index  = [i+2 for i in range(0,len(data_pos)-1,3)]
            data_v = np.copy(data_pos)
            data_v = np.delete(data_v,index)

        if (frame_index != 0):
            data_pos = self._configure_position(data_pos, frame_index, heigth)
            data_pos,data_v = self._smoother(data_pos, frame_index, heigth)

            self.previous_f[heigth] = []
            for i in range(0, self.nr_contours * self.nr_features, self.nr_features):
                self.previous_f[heigth].append([data_pos[i], data_pos[i + 1]])

        self.track_frame[heigth] = frame_index



        return data_pos,data_v

    def _configure_position(self, data_np, frame_index, heigth):

        data = data_np
        prev_f = self.previous_f[heigth]
        data_r = self._mapping(data, prev_f, frame_index)

        return data_r

    def _mapping(self, data, prev_f, frame_index):
        dict_mapping = {}

        data_tf = data

        data_ = []
        prev_data_ = prev_f
        for i in range(0, self.nr_contours * self.nr_features, self.nr_features):
            data_.append([data[i], data[i + 1]])

        data = sorted(data_, reverse=True)
        prev_data = prev_data_

        sum_d = 0

        for i in range(len(prev_data)):
            if (sum(data[i]) != 0):
                sum_d += 1

        if (sum_d != 0):

            distance_mat = np.zeros((self.nr_contours, self.nr_contours))

            for i in range(self.nr_contours):
                for j in range(self.nr_contours):
                    if (data[i][0] != 0 and data[i][1] != 0 and prev_data[j][0] != 0 and prev_data[j][1] != 0):
                        distance_mat[i][j] = np.sqrt(np.power(data[i][0] - prev_data[j][0], 2) +
                                                     np.power(data[i][1] - prev_data[j][1], 2))
                    else:
                        distance_mat[i][j] = 100000.

            dict_mapping = {}

            for i in range(self.nr_contours):
                argmin_a = np.zeros(self.nr_contours, dtype='uint8')
                val_a = np.zeros(self.nr_contours)

                for j in range(self.nr_contours):
                    argmin_a[j] = int(np.argmin(distance_mat[j]))
                    val_a[j] = distance_mat[j][argmin_a[j]]

                argmin = int(np.argmin(val_a))
                dict_mapping[argmin] = argmin_a[argmin]

                for j in range(self.nr_contours):
                    distance_mat[j][argmin_a[argmin]] = 99999999999.
                for j in range(self.nr_contours):
                    distance_mat[argmin][j] = 99999999999.



        else:
            for i in range(self.nr_contours):
                dict_mapping[i] = i

        # if(frame_index < 35 and frame_index > 30):
        #     print('MAPPING AND RETURN ARRAY', frame_index)
        #     print(dict_mapping)
        #     print('PREVIOUS: ', self.previous_f)
        #     print('INCOMING DATA', data)

        data_np = []

        for i in range(0, len(data_tf), self.nr_features):
            tmp = [data_tf[i], data_tf[i + 1], data_tf[i + 2]]
            data_np.append(tmp)

        data_np = sorted(data_np, reverse=True)

        # print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        # print(frame_index)
        # print('NEW DATA: ', data)
        # print('PREV DATA:', prev_data)
        # print(dict_mapping)

        data_r = np.zeros(self.nr_features * self.nr_contours)
        for i in range(len(data_np)):
            index = dict_mapping[i]
            data_r[self.nr_features * index:self.nr_features * index + self.nr_features] = data_np[i]

        # print(data_np)
        # print(data_r)
        # print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        #
        return data_r

    def _smoother(self, data, frame_index, heigth):
        if (frame_index != self.track_frame[heigth] and heigth == 60):
            self.change_copy = np.copy(self.change)

        for i in range(self.nr_contours):
            X = data[3 * i]
            y = data[3 * i + 1]

            if (self.track_frame[heigth] != frame_index):

                X_prev = self.previous_f[heigth][i][0]
                y_prev = self.previous_f[heigth][i][1]

                if (X_prev == 0 and y_prev == 0 and X != 0 and y != 0):
                    self.state[heigth][3 * i] = X
                    self.state[heigth][3 * i + 1] = y
                    self.state[heigth][3 * i + 2] = 0

            if (X != 0 and y != 0):
                data[3 * i] = data[3 * i] - self.state[heigth][3 * i] + 0.1
                data[3 * i + 1] = data[3 * i + 1] - self.state[heigth][3 * i + 1] + 0.1
                data[3 * i + 2] = data[3 * i + 2] - self.state[heigth][3 * i + 2]

        index = [i + 2 for i in range(0, len(data) - 1, 3)]
        data_v = np.copy(data)
        data_v = np.delete(data_v, index)

        for i in range(self.nr_contours):
            X = data[3 * i]
            y = data[3 * i + 1]
            if (frame_index != 0):
                if (X != 0 and y != 0):

                    if (frame_index != self.track_frame):
                        data_v[2 * i] = data_v[2 * i] - self.change[heigth][3 * i]
                        data_v[2 * i + 1] = data_v[2 * i + 1] - self.change[heigth][3 * i + 1]
                    else:
                        data_v[2 * i] = data_v[2 * i] - self.change_copy[heigth][3 * i]
                        data_v[2 * i + 1] = data_v[2 * i + 1] - self.change_copy[heigth][3 * i + 1]

            if (self.track_frame != frame_index):

                self.change[heigth][3 * i] = data[3 * i]
                self.change[heigth][3 * i + 1] = data[3 * i + 1]
                self.change[heigth][3 * i + 2] = data[3 * i + 2]

        return data,data_v


    def _config_prev(self):
        for heigth in self.resolution:
            self.previous_f[heigth] = np.zeros((self.nr_contours,2))
            self.state[heigth]      = np.zeros(self.nr_contours * 3)
            self.change[heigth]     = np.zeros(self.nr_contours * 3)

            self.track_frame[heigth] = 10000


class scaler():

    def __init__(self):
        pass

    def main(self,df,scaler_p,scaler_v):
        df =  apply_by_multiprocessing(df, self._transform_scaler,
                                                      scaler =scaler_p,
                                                      name   = 'data_p',
                                                      axis   = 1,
                                                      workers= 2)
        df = apply_by_multiprocessing(df, self._transform_scaler,
                                           scaler=scaler_v,
                                           name='data_v',
                                           axis=1,
                                           workers=2)


        return df




    def _train_scaler(self, args):
        scaler = MinMaxScaler(feature_range=(0, 1))

        df     = args[0]
        key    = args[1]

        data   = np.concatenate(list(df[key]))

        scaler.fit(data)

        return scaler

    def _transform_scaler(self,row,scaler=None,name=None):



        row[name] = scaler.transform(row[name])

        return row


class PCA_():

    def __init__(self,dict_c):
        self.dict_c = dict_c

    def main(self,df,path):


        PCA_mod    = decomposition.PCA(n_components=self.dict_c['PCA_components'])

        data_v     = np.concatenate(list(df['data_v']))
        data_p     = np.concatenate(list(df['data_p']))
        data       = np.concatenate([data_p,data_v], axis = 1)

        PCA_mod.fit(data)

        self.save_POV(PCA_mod,path)

        df = df.apply(self.transform_PCA,model = PCA_mod,axis=1)


        return df


    def transform_PCA(self,row,model = None):

        data             = np.concatenate([row['data_p'],row['data_v']], axis = 1)
        row['PCA']       = model.transform(data)

        return row

    def save_POV(self,PCA_mod,path):


        POV = PCA_mod.explained_variance_ratio_
        cum = PCA_mod.explained_variance_ratio_.cumsum()

        plt.plot(POV, label = 'POV')
        plt.plot(cum, label = 'cumsum POV')
        plt.legend()
        plt.title('Explained variability per PCA component')
        plt.xlabel('percentage of variability explained')
        plt.ylabel('percentage')
        plt.savefig(path+"/POV.png")


if __name__ == '__main__':
    pass

