import matplotlib.pyplot as plt
from   matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from   matplotlib.figure import Figure
import numpy as np
plt.style.use('ggplot')

import cv2
class plot_Tool():

    def __init__(self):
        pass

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


    def _config_f(self,feature):

        height  = feature%12
        contour = int(height/3)


        return contour









