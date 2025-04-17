import logging
from pathlib import Path
import os
from AlignAIR.PostProcessing.Steps.airr_finalization_and_packaging_steps import AIRRFinalizationStep
from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.correct_likelihood_for_genotype_step import GenotypeBasedLikelihoodAdjustmentStep
from AlignAIR.PostProcessing.Steps.finalization_and_packaging_steps import FinalizationStep
from AlignAIR.PostProcessing.Steps.germline_alignment_steps import AlleleAlignmentStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
from AlignAIR.PostProcessing.Steps.translate_to_imgt_step import TranslationStep
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
from AlignAIR.Step.Step import Step
from types import SimpleNamespace
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
def process_args (args):
    config = None

    if args.mode == 'cli':
        config = args

    return config
dsf = Path(r'C:\Users\tomas\Desktop\AlignAIRR\tests\sample_HeavyChain_dataset.csv')

if __name__ == '__main__':
    target_seq = 'GAGCTCTGGGAGAGGAGCCCAGCACTAGAAGTCGGCGGTGTTTCCATTCGGTGATCAGCACTGAACACAGAGGACTCACCATGGAGTTTGGGCTGAGCTGGGTTTTCCTCGTTGCTCTTTTAAGAGGTGTCCAGCCTGTGCAGCGTCTGGATTCACCTTCAGTAGTTATGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTGNCAGTTATATGGTATGATGGAAGTAATAAATACTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGGGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGATCTGAGTGCCGGATACAGCTATGCCTGTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAGGGAGTGCATCCGCCCCAACCCTTTTCCC'

    # save mock fasta file
    records = [
        SeqRecord(Seq(target_seq), id=f"test_seq_{i+1}", description="Copy of test sequence")
        for i in range(3)
    ]
    mock_fasta_path = Path(r'C:\Users\tomas\Desktop\AlignAIRR\tests\test_seq_.fasta')
    SeqIO.write(records, mock_fasta_path, "fasta")
    #
    import pandas as pd
    table = pd.read_table('C:/Users/tomas/Downloads/test_seq_test_seq__alignairr_results.csv', sep=',')
    records = [
        SeqRecord(Seq(seq), id=f"test_seq_{i + 1}", description="Copy of test sequence")
        for i,seq in enumerate(table['sequence'])
    ]
    mock_fasta_path = Path(r'C:\Users\tomas\Desktop\AlignAIRR\tests\test_seq_.fasta')
    SeqIO.write(records, mock_fasta_path, "fasta")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PipelineLogger')
    # mount logger to all step objects
    Step.set_logger(logger)
    # Parse command line arguments
    args = mock_args = SimpleNamespace(
        mode='cli',
        config_file=None,
        model_checkpoint='C:/Users/tomas/Desktop/AlignAIRR/tests/AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2',
        save_path=str(mock_fasta_path.stem),
        chain_type='heavy',
        sequences=str(mock_fasta_path),
        lambda_data_config='D',
        kappa_data_config='D',
        heavy_data_config='D',
        max_input_size=576,
        batch_size=2048,
        v_allele_threshold=0.1,
        d_allele_threshold=0.1,
        j_allele_threshold=0.1,
        v_cap=3,
        d_cap=3,
        j_cap=3,
        translate_to_asc=True,
        fix_orientation=True,
        custom_orientation_pipeline_path=None,
        custom_genotype=None,
        save_predict_object=False,
        airr_format=False,
        finetuned_model_params_yaml=None
    )

    # Process arguments based on mode
    config = process_args(args)

    predict_object = PredictObject(config, logger=logger)

    steps_dict = {
        "Load_Config": ConfigLoadStep("Load Config"),
        "Get_File_Name": FileNameExtractionStep("Get File Name"),
        "Count_Samples_in_File": FileSampleCounterStep("Count Samples in File"),
        "Load_Models": ModelLoadingStep("Load Models"),
        "Process_and_Predict_Batches": BatchProcessingStep("Process and Predict Batches"),
        "Clean_Up_Raw_Prediction": CleanAndArrangeStep("Clean Up Raw Prediction"),
        "Adjust_Likelihoods_for_Genotype": GenotypeBasedLikelihoodAdjustmentStep("Adjust Likelihoods for Genotype"),
        "Correct_Segmentations": SegmentCorrectionStep("Correct Segmentations"),
        "Apply_Max_Likelihood_Threshold_to_Distill_Assignments": MaxLikelihoodPercentageThresholdApplicationStep("Apply Max Likelihood Threshold to Distill Assignments"),
        "Align_Predicted_Segments_with_Germline": AlleleAlignmentStep("Align Predicted Segments with Germline")
    }
    if config.airr_format:
        steps_dict["Finalize_Results"] = AIRRFinalizationStep("Finalize Results")
    else:
        steps_dict["Translate_ASC's_to_IMGT_Alleles"] = TranslationStep("Translate ASC's to IMGT Alleles")
        steps_dict["Finalize_Results"] = FinalizationStep("Finalize Results")





    predict_object = steps_dict["Load_Config"].execute(predict_object)
    predict_object = steps_dict["Get_File_Name"].execute(predict_object)
    predict_object = steps_dict["Count_Samples_in_File"].execute(predict_object)
    predict_object = steps_dict["Load_Models"].execute(predict_object)
    predict_object = steps_dict["Process_and_Predict_Batches"].execute(predict_object)
    predict_object = steps_dict["Clean_Up_Raw_Prediction"].execute(predict_object)
    predict_object = steps_dict["Adjust_Likelihoods_for_Genotype"].execute(predict_object)
    predict_object = steps_dict["Correct_Segmentations"].execute(predict_object)
    predict_object = steps_dict["Apply_Max_Likelihood_Threshold_to_Distill_Assignments"].execute(predict_object)
    predict_object = steps_dict["Align_Predicted_Segments_with_Germline"].execute(predict_object)

    if config.airr_format:
        predict_object = steps_dict["Finalize_Results"].execute(predict_object)


    predict_object = steps_dict["Translate_ASC's_to_IMGT_Alleles"].execute(predict_object)
    predict_object = steps_dict["Finalize_Results"].execute(predict_object)


    # remove mock fasta file
    if os.path.exists(mock_fasta_path):
        os.remove(mock_fasta_path)

    saved_name = 'test_seq_test_seq__alignairr_results.csv'
    # remove the file if it exists
    # if os.path.exists(saved_name):
    #     os.remove(saved_name)


    print('Done')

#
# model_params = {'max_seq_length': 576, 'v_allele_count': 198, 'd_allele_count': 34, 'j_allele_count': 7}
# model = HeavyChainAlignAIRR(**model_params)
# trainer = Trainer(
#     model=model,
#     max_seq_length = model_params['max_seq_length'],
#     epochs=1,
#     batch_size=32,
#     steps_per_epoch=1,
#     verbose=1,
# )
# MODEL_CHECKPOINT = './AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
# trainer.load_model(MODEL_CHECKPOINT)
#
# # Trigger model building
# dummy_input = {
#     "tokenized_sequence": np.zeros((1, model_params['max_seq_length']), dtype=np.float32),
# }
# _ = trainer.model(dummy_input)  # Ensures the model builds and all layers are initialized
#
# prediction_Dataset = PredictionDataset(max_sequence_length=576)
# target_seq = 'CAGGTGCAGCTACAGCAGTGGGGCGCAGGACTGTTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTCAGTGGTTACTACTGGAGCTGGATCCGCCAGCCCCCAGGGAAGGGGCTGGAGTGGATTGGGGAAATCAATCATAGTGGAAGCACCAACTACAACCCGTCCCTCAAGAGTCGAGTCACCATATCAGTAGACACGTCCAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACCGCCGCGGACACGGCTGTGTATTACTGTGCGAGAGGCCGGCTGGCTGGAACGGCCCTCTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG'
#
# es = prediction_Dataset.encode_and_equal_pad_sequence(target_seq)['tokenized_sequence']
#
# predicted = trainer.model.predict({'tokenized_sequence':np.vstack([es])})
#
#



#
# seqs = pd.read_csv(dsf)
# encoded,pads  = trainer.train_dataset.encode_and_pad_sequences(seqs.sequence)
# print(encoded)
# pred = trainer.model.predict({'tokenized_sequence':np.vstack(encoded)})
# with open(r'C:\Users\tomas\Desktop\temp.pkl','wb') as h:
#     import pickle
#     pickle.dump((pred,pads,seqs),h)