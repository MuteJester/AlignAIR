

class AIRFormatColumnsManager:
    def __init__(self):
        self.required_columns = [
            'sequence_id',
            'sequence',
            'rev_comp',
            'productive',
            'v_call',
            'd_call',
            'j_call',
            'sequence_alignment',
            'germline_alignment',
            'junction',
            'junction_aa',
            'v_cigar',
            'd_cigar',
            'j_cigar',
        ]

    def get_junction_amino_acids(self,row):
        pass #TODO


    def convert_row(self,row):
        """
        This method will get a row from the pre-processed output of the AlignAIR with all the columns provided
        by the AlignAIR, and use them to derive and add the additional AIR format columns.
        This function should return a new row that will be replaced in the calling functions DataFrame
        """
        # TODO: implement the logic here!
        pass