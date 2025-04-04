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

class MutationRateSummaryPlotStep(Step):
    def execute(self, predict_object):
        self.log("Generating Model Mutation Rate Regression Figure")
        plt.figure(figsize=(20,11))
        sns.regplot(x=predict_object.groundtruth_table['mutation_rate'],
                        y=predict_object.results['cleaned_data']['mutation_rate'],
                    line_kws={'color':'tab:red'})
        plt.xlabel('Ground Truth Mutation Rate')
        plt.ylabel('Predicted Mutation Rate')
        plt.grid(lw=2,ls=':')

        plt.savefig(predict_object.script_arguments.save_path+'AlignAIR_MutationRate_Figure.pdf', dpi=200, bbox_inches='tight')

        return predict_object
