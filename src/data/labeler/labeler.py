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

class labeler():

    def __init__(self,path,list_names):
        self.path       = path
        self.list_names = list_names
        self.pause      = False

        self.label      = None
        self.df         = None
        self.len_df     = None

        self.track_k    = None
        self.track_i    = None
        self.track_rmt  = None

    def play_videos(self):
        i = 0
        j = 0
        k = 0
        while (1):
            # print(self.df.head()):
            try:
                i, j,k  = self._control_ijk(i, j,k)

                dict_ ,passages  = self.get_json(i,k)
                path    = self.path+self.list_names[k]+ '/visualized/' +  dict_['frames'][j]

                img     = cv2.imread(path, -1)
                img     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img     = cv2.resize(img,(600,800))
                self.write_data(img,i,j,k,dict_)
                self.save_json(i,k,self.label,passages)


                cv2.imshow('frame', img)
                time.sleep(0.05)

            except Exception as e:

                print(e)
                if(self.pause == True and self.track_rmt == False):
                    shutil.rmtree(self.path+self.list_names[k])
                    self.track_rmt = True

            # Controls GUI
            if (self.pause == False):
                j += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            i, j,k = self._control_videos(key, i, j,k)
        cv2.destroyAllWindows()

    def save_json(self,i,k,label,passages):
        if(label != None):
            passages['passages'][i]['label']['anomaly'] = label
            path = self.path +self.list_names[k]+ '/passages.json'

            df   = pd.DataFrame(passages)
            df.to_json(path)

    def get_json(self,i,k):
        path        = self.path +self.list_names[k]+ '/passages.json'
        passages    = pd.read_json(path)

        try:
            dict_ = {'label': passages['passages'][i]['label']['anomaly'],
                     'frames': passages['passages'][i]['frames'],
                     'name' :  self.list_names[k]}

        except Exception as e:

            passages['passages'][i]['label']['anomaly'] = None
            passages['passages'][i]['label']['personCountDelta'] = None

            print('lol' +str(e))
            dict_ = {'label': 'und',
                     'frames': passages['passages'][i]['frames'],
                     'name':  self.list_names[k]}


        return dict_,passages

    def write_data(self,frame,i,j,k,dict_):
        scenario = self.list_names[k]
        label    = dict_['label']

        string   = ' scenerio: '+str(scenario)
        cv2.putText(frame,
                    string,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

        string = str(k)+'/'+str(len(self.list_names))+' movie id: '+str(i)+'/'+str(len(self.df))+' Frame: '+str(j)+'/'+str(len(dict_['frames']))+' Label: '+str(label)
        cv2.putText(frame,
                    string,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

    def _control_videos(self,key,i,j,k):

            if key == ord('r'):
                if(i< self.len_df-1):
                    i += 1
                else:
                    i  = 0

                j  = 0

            if key == ord('e'):
                if i == 0:
                    i = self.len_df - 1
                else:
                    i -= 1

                j = 0

            if key == ord('d'):
                k += 1
                j  = 0

            if key == ord('s'):
                if k == 0:
                    k = len(self.list_names) - 1
                else:
                    k -= 1

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

            return i,j,k

    def _control_ijk(self,i,j,k):
        if(k != self.track_k):
            self.df        = pd.read_json(self.path+self.list_names[k]+'/passages.json')
            self.len_df    = len(self.df)
            self.track_k   = k
            self.track_rmt = False
            self.label     = None
            i              = 0


        if(i != self.track_i):
            self.label   = None
            self.track_i = i

        if (i >= self.len_df):
            i = 0
        if (j >= len(self.df['passages'][i]['frames'])):
            j = 0


        return i,j,k

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



    path = './data/raw/conf_FS/hallway/'

    list_names = os.listdir(path)

    lab = labeler(path,list_names)
    lab.play_videos()
    #
    # path_in  = './data/raw/conf_FS/'
    # path_out = './data/raw/configured_raw/'
    #
    # relocate(path_in,path_out).main()