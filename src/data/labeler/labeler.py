import shutil
from dotenv import find_dotenv, load_dotenv
import os
import sys

load_dotenv(find_dotenv())
PATH_P = os.environ['PATH_P']
os.chdir(PATH_P)
sys.path.insert(0, PATH_P)

import cv2
import time
import pandas as pd
from src.data.dimensionality_reduction.HCF.preprocessing import get_df
import matplotlib.pyplot as plt
class restructure_FS():

    def __init__(self):
        self.path_in  = './data/raw/new_data/'
        self.path_out = './data/raw/conf_FS/'

    def main(self):
        venues = os.listdir(self.path_in)


        for venue in venues:
            path_out = self.path_out + venue + '/'
            if('testroom' not in venue):
                path_v    = self.path_in+venue+'/'
                scenarios = os.listdir(path_v)

                for scenario in scenarios:
                    if('labels' not in scenario):

                        path_i_s = path_v + scenario +'/'
                        path_o_s = path_out + scenario +'/'

                        self.configure_scenario(path_o_s)
                        self.scenario(path_i_s,path_o_s)

    def remove_white_space(self):
        venues = os.listdir(self.path_out)

        for venue in  venues:
            path_v = self.path_out + venue + '/'
            scenarios = os.listdir(path_v)

            for scenario in scenarios:
                src = path_v + scenario + '/'
                dst = src.replace(' ','')
                os.rename(src, dst)

    def configure_scenario(self,path_o_s):

        if (os.path.exists(path_o_s)==False):
            os.mkdir(path_o_s)

        path_o_s_d  = path_o_s + 'depth-references/'
        if (os.path.exists(path_o_s_d)==False):
            os.mkdir(path_o_s_d)

        path_o_s_dv = path_o_s_d +'visualized/'
        if (os.path.exists(path_o_s_dv)==False):
            os.mkdir(path_o_s_dv)


        path_o_r    = path_o_s + 'raw/'
        if (os.path.exists(path_o_r)==False):
            os.mkdir(path_o_r)

        path_o_v    = path_o_s + 'visualized/'
        if (os.path.exists(path_o_v)==False):
            os.mkdir(path_o_v)

    def scenario(self, path_i_s,path_o_s):

        path_i_depth   = path_i_s + 'depth-references/'
        path_o_depth   = path_o_s + 'depth-references/'

        list_d         = os.listdir(path_i_depth)

        for string in list_d:
            if('visual' not in string):
                path_i_depth_r = path_i_depth   + string
                path_o_depth_r = path_o_depth   + string

                shutil.copy(path_i_depth_r,path_o_depth_r)
            else:
                path_i_depth_r = path_i_depth +'visualized/'
                path_o_depth_r = path_o_depth +'visualized/'

                listd2        = os.listdir(path_i_depth_r)

                for string2 in listd2:
                    path_i_depth_rv = path_i_depth_r + string2
                    path_o_depth_rv = path_o_depth_r + string2

                    shutil.copy(path_i_depth_rv, path_o_depth_rv)



        path_i_pass  = path_i_s + 'labels.json'
        path_o_pass  = path_o_s + 'passages.json'

        shutil.copy(path_i_pass ,path_o_pass)

        path_seq_ir  = path_i_s  + 'sequences/'
        path_seq_iv  = path_i_s  + 'sequences/visualized/'

        raw           = os.listdir(path_seq_ir)

        if(len(raw) != 0):
            visualized = os.listdir(path_seq_iv)

            for  raw_img in raw:

                if('visualized' not in raw_img):
                    src       = path_seq_ir+raw_img
                    dst       = path_o_s+ 'raw/'+raw_img
                    shutil.copy(src,dst)

            for vis_img in visualized:

                if('depth-openni' not in vis_img):

                    src = path_seq_iv + vis_img
                    dst = path_o_s + 'visualized/' + vis_img
                    shutil.copy(src, dst)

        else:
            shutil.rmtree(path_o_s)

    def remove_trees(self,path,list_names):

        names = os.listdir(path)

        for name in names:
            if(name not in list_names):
                path_r = path+name
                shutil.rmtree(path_r)



class add_segmentation():

    def __init__(self,path):

        self.path       = path
        self.list_names = os.listdir(path)
        self.df         = get_df(self.path).get_df_data(self.list_names)
        print(len(self.df))

    def main(self):
        sum_ = 0
        for i in range(len(self.list_names)):


            path = self.path + self.list_names[i] + '/passages.json'
            passages = pd.read_json(path)

            for j in range(len(passages['passages'])):
                passages['passages'][j]['label']['segmentation'] = 'None'


            passages.to_json(path)


class anomaly_segmenter():

    def __init__(self,path,list_names):
        self.path       = path
        self.list_names = list_names
        self.pause      = False

        self.df_anom    = get_df(path).get_df_data(list_names)
        self.df_anom    = self.df_anom[self.df_anom['label'] == True]


        self.df         = None
        self.len_df     = None

        self.track_i      = None
        self.track_rmt    = None
        self.segmentation = None
        self.label        = None

        self.save_b       = False
    def play_videos(self):

        i = 0
        j = 0
        while (1):
            # print(self.df.head()):
            i, j  = self._control_ijk(i, j)


            path    = self.path+self.df_anom['name'].iloc[i]+ '/visualized/' +  self.df_anom['frames'].iloc[i][j]
            img     = cv2.imread(path, -1)
            img_c   = img.copy()
            img     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img     = cv2.resize(img,(600,800))


            path_p           = self.path + self.df_anom['name'].iloc[i] + '/passages.json'
            dict_ ,passages  = self.get_json(i,path_p)
            print(img_c.shape)
            print(dict_['segment'])
            print(self.save_b)
            if (self.save_b == True):
                cv2.imwrite('picture_' + dict_['segment'] + '.PNG', img_c)


            self.write_data(img,i,j,dict_)
            # self.save_json(i,self.label,self.segmentation,dict_,passages,path_p)


            cv2.imshow('frame', img)
            time.sleep(0.04)



            # Controls GUI
            if (self.pause == False):
                j += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            i, j = self._control_videos(key, i, j)
        cv2.destroyAllWindows()

    def save_json(self,i,label,segmentation,dict_,passages,path_p):
        if(label != None):
            passages['passages'][dict_['index']]['label']['anomaly'] = label


        if (segmentation != None):
            passages['passages'][dict_['index']]['label']['segmentation'] = segmentation

        passages.to_json(path_p)

    def get_json(self,i,path):
        passages    = pd.read_json(path)

        for index,passage in enumerate(passages['passages']):

            if(passage['frames'] == self.df_anom.iloc[i]['frames']):

                dict_ = {'label' : passage['label']['anomaly'],
                         'frames': passage['frames'],
                         'segment':passage['label']['segmentation'],
                         'name'  : self.df_anom['name'].iloc[i],
                         'index' : passage['id']}




        return dict_,passages

    def write_data(self,frame,i,j,dict_):
        scenario = self.df_anom['name'].iloc[i]
        label    = dict_['label']
        segment  = dict_['segment']

        if(type(segment) == list):
            string  = ''
            for seq in segment:
                if(seq == None):
                    string += str(seq)+'/ '
                else:
                    string += seq+'/ '

            segment = string
        else:

            segment = str(segment)

        string   = ' scenerio: '+str(scenario)
        cv2.putText(frame,
                    string,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)
        string = str(i)+'/'+str(len(self.df_anom)) +' Frame: '+str(j)+'/'+str(len(dict_['frames']))+' Label: '+str(label)+' segment: '+segment
        cv2.putText(frame,
                    string,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

    def _control_videos(self,key,i,j):

            if key == ord('r'):
                if(i< len(self.df_anom)-1):
                    i += 1
                else:
                    i  = 0

                j  = 0

            if key == ord('e'):
                if i == 0:
                    i = len(self.df_anom) - 1
                else:
                    i -= 1

                j = 0


            if key == ord('p'):
                self.pause = True

            if key == ord('o'):
                self.pause = False

            if key == ord('t'):
                self.label = True

            if key == ord('f'):
                self.label = False

            if key == ord('k'):
                j -= 1

            if key == ord('l'):
                j += 1

            if key == ord('t'):
                self.label = True

            if key == ord('f'):
                self.label = False
            
            if key == ord('x'):
                self.segmentation = 'boven'

            if key == ord('c'):
                self.segmentation = 'e'

            if key == ord('v'):
                self.segmentation = 'object'

            if key == ord('b'):
                self.segmentation = 'gooien'

            if key == ord('n'):
                self.segmentation = 'muren'

            if key == ord('z'):
                self.segmentation = 'sneaky'
                
            if key == ord('m'):
                self.segmentation = 'None'

            if key == ord('1'):
                self.save_b = True

            if key == ord('2'):
                self.save_b = False
 
            return i,j

    def _control_ijk(self,i,j):
        if(i != self.track_i):

            self.df               = pd.read_json(self.path+self.df_anom['name'].iloc[i]+'/passages.json')
            self.len_df           = len(self.df)
            self.track_i          = i
            self.track_rmt        = False
            self.label            = None
            self.segmentation     = None


        if (i >= len(self.df_anom)):
            i = 0
        if (j >= len(self.df_anom['frames'].iloc[i])):
            j = 0


        return i,j


class relocate():

    def __init__(self,path_in,path_out):

        self.path_i = path_in
        self.path_o = path_out

    def main(self):

        list_names = os.listdir(self.path_i)

        for name in list_names:

            path_v = self.path_i + name + '/'

            scenarios = os.listdir(path_v)

            for scenario in scenarios:
                try:
                    src = path_v + scenario
                    dst = self.path_o + scenario

                    shutil.copytree(src,dst)
                except:
                    pass


if __name__ == '__main__':
    # FS_ = restructure_FS()
    # FS_.remove_white_space()
    # FS_.main()
    # list_names = ['Scenario_2017-09-11 14:33:19',
    #               'Scenario_2017-09-04 17:18:27',
    #               'Scenario_2017-09-04 17:21:55',
    #               'Scenario_2017-09-11 14:22:00']

    # path = './data/raw/configured_raw/'
    # FS_.remove_trees(path,list_names)



    path = './data/raw/configured_raw/'

    # add_segmentation(path).main()



    list_names = os.listdir(path)
    #
    lab = anomaly_segmenter(path,list_names)
    lab.play_videos()
    #
    # path_in  = './data/raw/conf_FS/'
    # path_out = './data/raw/configured_raw/'
    #
    # relocate(path_in,path_out).main()