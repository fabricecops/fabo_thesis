
import multiprocessing as mp
import time
from src.data.dimensionality_reduction.HCF.preprocessing import path_Generation as Path_gen
from src.visualization.vis_LSTM_p.utilities.conf_data import *
from src.visualization.vis_LSTM_p.utilities.plt_data import *


class main_visualize(conf_data,plot_Tool,Path_gen):

    def __init__(self,dict_c):
        self.height          = 50

        conf_data.__init__(self,dict_c)
        plot_Tool.__init__(self)
        Path_gen.__init__(self,self.df,self.dict_c)

        self.feature,_ = self._choose_feature()

        self.track_f    = 1000
        self.track_i    = 1000

        self.work_queue = mp.Queue()
        self.done_queue = mp.Queue()
        self.workers    = 6
        self.processes  = []

        self.array_p    = []

        self.pause      = False

        self.threshold       = 0.
        self.track_threshold = None
        self.plot_mode   = dict_c['plot_mode']

    def play_videos(self):
        i     = 0
        j     = 0
        while (1):

            i,j          = self._control_ij(i,j)
            boolean      = self._configure_boolean(i,self.feature)
            self._multiprocessing(i,self.feature,boolean,j)
            self._configure_array(j)
            img          = self.array_p[j]

            frame_pd     = self.configure_frame(self.df,i,j, self.height)
            frame_fin    = np.concatenate((frame_pd,img),axis = 0)

            # self._write_auc( frame_fin,i)

            cv2.imshow('frame', frame_fin)
            time.sleep(0.1)

            # Controls GUI
            if(self.pause == False):
                j += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            i,j =self._control_videos(key, i, j)
        cv2.destroyAllWindows()

    def _write_auc(self,frame,i):
        AUC = self.AUC


        string_TP            = str(len(self.df_true[self.df_true['error_tm']>self.threshold]))+'/'+str(len(self.df_true))
        string_TN            = str(len(self.df_false[self.df_false['error_tm']<=self.threshold]))+'/'+str(len(self.df_false))


        error = self.df.iloc[i]['error_tm']


        if (error > self.threshold):
            p = 'Anomaly'
        else:
            p = 'Normal'

        string = 'AUC: '+str(round(np.abs(AUC),2))+\
                 ' TH: '+str(round(self.threshold,3))+\
                 ' TF: '+str(string_TN)+\
                 ' TP: '+  str(string_TP)+\
                 ' Prediction: '+p+\
                 ' E: '+str(error)


        cv2.putText(frame,
                    string,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

    def _multiprocessing(self,i,feature,boolean,j):

        if(boolean == False):
            for p in self.processes:
                if(p.is_alive()):
                    p.terminate()

            self.processes  = []
            self.done_queue = mp.Queue()
            self.work_queue = mp.Queue()


            j_mp = [j_ for j_ in range(j,len(self.df.iloc[i]['frames']))]
            j_mp.extend(j_ for j_ in range(j))

            for j in j_mp:
                self.work_queue.put(j)

            self.work_queue.put('STOP')

            for w in range(self.workers):
                p        = mp.Process(target=self._worker, args=(i, feature))
                p.daemon = True
                p.start()
                self.processes.append(p)
        boolean_count = len(self.df.iloc[i]['frames']) == self._count_finished_frames()

        if(boolean == True and boolean_count):
            for p in self.processes:
                if(p.is_alive()):
                    p.terminate()

    def _worker(self, i, feature):
        # try:
            for j in iter(self.work_queue.get,'STOP'):
                img      = self._get_plot(i,j,self.feature)
                tuple_   = (j,img)

                self.done_queue.put(tuple_)
        # except Exception as e:
        #     print(e)

            return True

    def _configure_array(self,j):
        while True:
            try:
                if(self.done_queue.empty() != True):

                    img_t = self.done_queue.get()
                    self.array_p[img_t[0]] = img_t[1]

                if(type(self.array_p[j])== np.ndarray):
                    break
                time.sleep(0.1)
            except Exception as e:
                print(e)

    def _configure_boolean(self,i,feature):

        boolean = True
        if(i != self.track_i):
            self.track_i = i
            boolean      = False
            self.array_p = [1000 for j in range(len(self.df.iloc[i]['frames']))]

        if(feature != self.track_f):
            self.track_f = feature
            boolean = False
            self.array_p = [1000 for j in range(len(self.df.iloc[i]['frames']))]


        return boolean

    def _count_finished_frames(self):
        count = 0

        for i in range(len(self.array_p)):
            if(type(self.array_p[i])== np.ndarray):
                count +=1

        return count

    def _get_plot(self,i,j,feature):


        if(self.plot_mode == 'feature'):

            output_A, pred_A, error_A = self.conf_pred_feat(i,feature)
            output_y, pred_y, error_y = self.conf_pred_feat(i,feature+1)
            output_x, pred_x, error_x = self.conf_pred_feat(i,feature+2)


            img = self.plot_3(j,feature, output_A, pred_A, error_A,output_y, pred_y, error_y,output_x, pred_x, error_x)

        elif self.plot_mode == 'dist':
            img = self.get_plot_error(self.df_true,self.df_false)

        elif self.plot_mode == 'error':
            img = self.get_plot_error_con(i,j,self.df)





        return img

    def _choose_feature(self):
        min_f   = int(self.height/10 * 12)
        max_f   = int(min_f+11)

        return min_f,max_f

    def _control_videos(self,key,i,j):

        if key == ord('r'):
            i += 1
            j  = 0

        if key == ord('e'):
            if i == 0:
                i = len(self.df) - 1
            else:
                i -= 1

            j = 0

        if key == ord('f'):
            self.height += self.dict_c['resolution']

            if (self.height > self.dict_c['max_h']):
                self.height = self.dict_c['max_h']

            min_f, _ = self._choose_feature()
            self.feature = min_f

        if key == ord('d'):

            self.height -= self.dict_c['resolution']
            if (self.height < self.dict_c['min_h']):
                self.height = self.dict_c['min_h']

            min_f, _ = self._choose_feature()
            self.feature = min_f
            self.feature = min_f

        if key == ord('v'):
            _, max_f = self._choose_feature()
            self.feature += 3
            if (self.feature > max_f):
                self.feature = max_f

        if key == ord('c'):
            min_f, _ = self._choose_feature()
            self.feature -= 3

            if (self.feature < min_f):
                self.feature = min_f

        if key == ord('p'):
            self.pause = True

        if key == ord('o'):
            self.pause = False

        if key == ord('t'):
            self.threshold -= 0.01

            if(self.threshold < 0.0):
                self.threshold = 0.0

        if key == ord('y'):
            self.threshold += 0.01

        if key == ord('k'):
            j -= 1

        if key == ord('l'):
            j += 1

        return i,j

    def _control_ij(self,i,j):
        if (i >= len(self.df)):
            i = 0
        if (j >= len(self.df.iloc[i]['frames'])):
            j = 0
        return i,j

if __name__ == '__main__':
    path_best = './models/bayes_opt/DEEP2/21/'
    dict_c = {
        'path': path_best,
        'mode': 'df_t_train',

        'path_dict': path_best + 'dict.p',

        'plot_mode': 'error'

    }
    vis = main_visualize(dict_c)
    vis.play_videos()
    vis = main_visualize(dict_c)
    vis.play_videos()

