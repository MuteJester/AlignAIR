from abc import abstractmethod,ABC
from scipy.stats import entropy

class Filter(ABC):
    @abstractmethod
    def test(self,**kwargs):
        raise NotImplementedError


class CallEntropy(Filter):
    def test(self,X,threshold):
        return False if entropy(X) > threshold else True

class SegmentIntersection(Filter):
    """
    Test whether there are intersections between segments, segments that is
    contained inside a different section should be
    """

    def test(self,X):
        for segment_col_name in ['v_start','v_end','j_start','j_end']:
            assert segment_col_name in X.index

