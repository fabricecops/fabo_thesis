import matplotlib.pyplot as plt
from   matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from   matplotlib.figure import Figure
import numpy as np
plt.style.use('ggplot')
from src.dst.outputhandler.pickle import pickle_load,pickle_save

import cv2
class plot_Tool():

    def __init__(self):
        pass

    def save_image(self,j,feature, output_A, pred_A, error_A,
                               output_y, pred_y, error_y,
                               output_x, pred_x, error_x,tracker):
        print('haha')
        data = j,feature, output_A, pred_A, error_A, output_y, pred_y, error_y, output_x, pred_x, error_x
        pickle_save('./plots/AFE/'+str(tracker)+'.p',data)




        contour = self._config_f(feature)
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(131)
        ax.plot(pred_A, label='prediction', color='blue', linewidth=3.3)
        ax.plot(output_A, label='output', color='green', linewidth=3.3)
        ax.plot(error_A, color='black', label='error')
        plt.plot([j], [output_A[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_A[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [error_A[j]], marker='o', markersize=3, color="black")

        plt.legend()
        plt.title('X with contour: ' + str(contour))
        plt.xticks(rotation=70)

        ax = plt.subplot(132)
        ax.plot(pred_x, label='prediction', color='blue', linewidth=3.3)
        ax.plot(output_x, label='output', color='green', linewidth=3.3)
        ax.plot(error_x, color='black', label='error')
        plt.plot([j], [output_x[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_x[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [error_x[j]], marker='o', markersize=3, color="black")

        plt.title('area')
        plt.xticks(rotation=70)

        ax = plt.subplot(133)
        ax.plot(pred_y, label='prediction', color='blue', linewidth=3.3)
        ax.plot(output_y, label='output', color='green', linewidth=3.3)
        ax.plot(error_y, color='black', label='error')
        plt.plot([j], [output_y[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_y[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [error_y[j]], marker='o', markersize=3, color="black")

        plt.title('dy')
        plt.xticks(rotation=70)
        plt.plot(error_A , color = 'black',     label = 'error')
        plt.axhline(0.003, color = 'red',       label = 'threshold')
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Sequence id')
        plt.savefig('./plots/plots/LSTM_error.png')

    def plot_3(self,j,feature, output_A, pred_A, error_A,
                               output_y, pred_y, error_y,
                               output_x, pred_x, error_x):

        contour = self._config_f(feature)
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(131)
        ax.plot(pred_A,  label = 'prediction', color = 'blue',linewidth=3.3)
        ax.plot(output_A,label = 'output'    , color = 'green',  linewidth=3.3)
        ax.plot(error_A , color = 'black',     label = 'error')
        plt.plot([j], [output_A[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_A[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [error_A[j]], marker='o', markersize=3, color="black")

        plt.legend()
        plt.title('X with contour: '+str(contour))
        plt.xticks(rotation=70)

        ax = plt.subplot(132)
        ax.plot(pred_x,  label = 'prediction', color = 'blue',linewidth=3.3)
        ax.plot(output_x,label = 'output'    , color = 'green',  linewidth=3.3)
        ax.plot(error_x , color = 'black',     label = 'error')
        plt.plot([j], [output_x[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_x[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [error_x[j]], marker='o', markersize=3, color="black")

        plt.title('area')
        plt.xticks(rotation=70)

        ax = plt.subplot(133)
        ax.plot(pred_y,  label = 'prediction', color = 'blue',linewidth=3.3)
        ax.plot(output_y,label = 'output'    , color = 'green',  linewidth=3.3)
        ax.plot(error_y , color = 'black',     label = 'error')
        plt.plot([j], [output_y[j]], marker='o', markersize=6, color="black")
        plt.plot([j], [pred_y[j]], marker='o', markersize  =6, color="black")
        plt.plot([j], [error_y[j]], marker='o', markersize =3, color="black")

        plt.title('dy')
        plt.xticks(rotation=70)
        # plt.plot(error_A , color = 'black',     label = 'error')
        # plt.axhline(0.003, color = 'red',       label = 'threshold')
        # plt.legend()
        # plt.ylabel('Error')
        # plt.xlabel('Sequence id')
        # plt.savefig('./plots/plots/LSTM_error.png')
        fig.canvas.draw()


        image  = np.array(fig.canvas.renderer._renderer)
        plt.close()

        image = cv2.resize(image, (960, 240))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def get_plot_error_con(self,i,j,df):
        fig = plt.figure(figsize=(16, 4))
        plt.plot(np.array(df['error_v'].iloc[i]), label='prediction', color='blue', linewidth=3.3)
        plt.plot([j],np.array(df['error_v'].iloc[i])[j], marker='o', markersize=6, color="black")

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer)
        image = np.array(fig.canvas.renderer._renderer)
        plt.close()

        image = cv2.resize(image, (960, 240))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def get_plot_error(self,df_true,df_false):

        fig = plt.figure(figsize=(16, 4))

        ax2 = plt.subplot(121)
        ax2.hist(df_true['error_tm'], label='True', color='red', alpha=0.5, bins=50, range=(0, 1))
        ax2.hist(df_false['error_tm'], label='False', color='green', alpha=0.5, bins=50, range=(0, 1))
        plt.legend()
        ax2 = plt.subplot(122)
        # ax2.hist(df_true['error_tm'], label='True', color='red', alpha=0.5, bins=50, range=(0, 10))
        ax2.hist(df_false['error_tm'], label='False', color='green', alpha=0.5, bins=50, range=(0, 10))
        plt.legend()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer)
        plt.close()

        image = cv2.resize(image, (960, 240))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def _config_f(self,feature):

        height  = feature%12
        contour = int(height/3)


        return contour









