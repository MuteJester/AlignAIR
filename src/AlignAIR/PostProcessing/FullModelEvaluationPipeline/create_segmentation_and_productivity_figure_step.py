from collections import defaultdict
from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold
from AlignAIR.Step.Step import Step
from sklearn.metrics import confusion_matrix

class SegmentationProductivitySummaryPlotStep(Step):

    def rmse(self,predicted,groundtruth):
        return np.sqrt(np.mean(( ( (predicted-groundtruth)**2 ) )))

    def get_segmentation_barplot_data(self,predict_object):
        seg_error_data = {'productive':{'aligner':[],'variable':[],'value':[]},
                          'non_productive':{'aligner':[],'variable':[],'value':[]}}


        for pstatus,flag in zip(['productive','non_productive'],[True,False]):
            productive_sample_mask = predict_object.groundtruth_table.productive == flag
            for gene in ['v','d','j']:
                for (pos,orn) in (zip(['start','end'],['3','5'])):
                    seg_error_data[pstatus]['aligner'].append('AlignAIR')
                    seg_error_data[pstatus]['variable'].append(f'{gene}{orn}')
                    seg_error_data[pstatus]['value'].append(self.rmse(predict_object.groundtruth_table[productive_sample_mask][f'{gene}_sequence_{pos}'],
                                                                      predict_object.results['corrected_segments'][f'{gene}_{pos}'][productive_sample_mask]))

        return seg_error_data

    def get_productive_accuracy_confusion_matrix(self,predict_object):
        predicted_productivity = predict_object.results['cleaned_data']['productive']>0.5

        cm = confusion_matrix(predict_object.groundtruth_table['productive'],
                              predicted_productivity,normalize='true')

        return cm




    def execute(self, predict_object):
        self.log("Generating Model Segmentation RMSE and Productivity Accuracy Summary Figure")
        seg_error_data = self.get_segmentation_barplot_data(predict_object)
        productivity_cm = self.get_productive_accuracy_confusion_matrix(predict_object)

        mosaic = """
                       AC
                       BB
                       """
        fig = plt.figure(layout="tight", figsize=(20, 15))
        axd = fig.subplot_mosaic(mosaic)

        sns.barplot(x='variable', y='value',
                    hue='aligner', data=seg_error_data['non_productive'], ax=axd['A'])
        axd['A'].set_xlabel('Position')
        axd['A'].set_ylabel('RMSE')
        axd['A'].grid(lw=2, ls=':', axis='y')
        #axd['A'].set_ylim(0, 4)
        axd['A'].set_title('Corrupted Sequences')
        axd['A'].legend_ = None

        sns.barplot(x='variable', y='value',
                    hue='aligner', data=seg_error_data['productive'], ax=axd['C'])
        axd['C'].set_xlabel('Position')
        axd['C'].set_ylabel('RMSE')
        axd['C'].grid(lw=2, ls=':', axis='y')
        #axd['C'].set_ylim(0, 4)
        axd['C'].set_title('Productive Sequences')
        axd['C'].legend(borderaxespad=1)

        # heatmaps

        sns.heatmap(productivity_cm, ax=axd['B'], annot=True, cmap='coolwarm',
                    fmt='.1%', lw=2, vmin=0, vmax=1, xticklabels=['Productive', 'Non\nProductive'],
                    yticklabels=['Productive', 'Non\nProductive'], cbar=False)
        axd['B'].set_title('AlignAIRR', fontweight="bold")

        # plt.tight_layout()

        # for x in axd:
        #     ax = axd[x]
        #     for line in ax.get_lines():
        #         line.set_path_effects([
        #             plt.matplotlib.patheffects.withStroke(linewidth=0)  # No outline
        #         ])

        # plt.savefig('poster_seg_prod.png',dpi=200,bbox_inches='tight')

        plt.savefig(predict_object.script_arguments.save_path+'AlignAIR_SegmentationProductivity_Figure.pdf', dpi=200, bbox_inches='tight')

        return predict_object
