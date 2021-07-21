import pandas as pd
import numpy as np
import torch

### port of https://github.com/bes-dev/mean_average_precision from numpy to pytorch

class MetricBase:
    """ Implements base interface for evaluation metrics."""
    def add(self, *args, **kwargs):
        """ Add sample to metric."""
        raise NotImplemented

    def value(self, *args, **kwargs):
        """ Get metric value."""
        raise NotImplemented

    def reset(self):
        """ Reset internal state of metric."""
        raise NotImplemented
        
        
class AdapterBase:
    """ Arguments Adapter for Metric.
    Arguments:
        metric_fn (MetricBase): metric function.
        value_config (dict): arguments of self..value(...) method.
    """
    def __init__(self, metric_fn, value_config=None):
        self.metric_fn = metric_fn
        self.value_config = value_config

    def add(self, preds, gt):
        """ Add sample to evaluation.
        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.
        """
        preds, gt = self._check_empty(preds, gt)
        preds = self._preds_adapter(preds)
        gt = self._gt_adapter(gt)
        return self.metric_fn.add(preds, gt)

    def value(self, *args, **kwargs):
        """ Evaluate Metric.
        Arguments:
            *args, **kwargs: metric specific arguments.
        Returns:
            metric (dict): evaluated metrics.
        """
        if self.value_config is not None:
            return self.metric_fn.value(**self.value_config)
        else:
            return self.metric_fn.value(*args, **kwargs)

    def reset(self):
        """ Reset stored data.
        """
        return self.metric_fn.reset()

    def _check_empty(self, preds, gt):
        """ Check empty arguments
        Arguments:
            preds (torch.tensor): predicted boxes.
            gt (torch.tensor): ground truth boxes.
        Returns:
            preds (torch.tensor): predicted boxes.
            gt (torch.tensor): ground truth boxes.
        """
        if preds.numel()==0:
            preds = torch.zeros((0, 6))
        if gt.numel()==0:
            gt = torch.zeros((0, 7))
        return preds, gt

    def _preds_adapter(self, preds):
        """ Preds adapter method.
        Should be implemented in child class.
        """
        raise NotImplemented
        
        
        
class AdapterDefault(AdapterBase):
    """ Default implementation of adapter class.
    """
    def _preds_adapter(self, preds):
        return preds

    def _gt_adapter(self, gt):
        return gt
    
    
class MetricBuilder:
    @staticmethod
    def get_metrics_list():
        """ Get evaluation metrics list."""
        return list(metrics_dict.keys())

    @staticmethod
    def build_evaluation_metric(metric_type, async_mode=False, adapter_type=AdapterDefault, *args, **kwargs):
        """ Build evaluation metric.
        Arguments:
            metric_type (str): type of evaluation metric.
            async_mode (bool): use multiprocessing metric.
            adapter_type (AdapterBase): type of adapter class.
        Returns:
            metric_fn (MetricBase): instance of the evaluation metric.
        """
        assert metric_type in metrics_dict, "Unknown metric_type"
        
        metric_fn = metrics_dict[metric_type](*args, **kwargs)
        
        #if not async_mode:
        #    metric_fn = metrics_dict[metric_type](*args, **kwargs)
        #else:
        #    metric_fn = MetricMultiprocessing(metrics_dict[metric_type], *args, **kwargs)
        return adapter_type(metric_fn)
      
      
class MeanAveragePrecision2d(MetricBase):
    """ Mean Average Precision for object detection.
    Arguments:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._init()

    def reset(self):
        """Reset stored data."""
        self._init()

    def add(self, preds, gt):
        """ Add sample to evaluation.
        Arguments:
            preds (torch.tensor): predicted boxes.
            gt (torch.tensor): ground truth boxes.
        Input format:
            preds: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        """
        assert preds.ndim == 2 and preds.shape[1] == 6
        assert gt.ndim == 2 and gt.shape[1] == 7
        class_counter = torch.zeros((1, self.num_classes), dtype=torch.int32)
        for c in range(self.num_classes):
            gt_c = gt[gt[:, 4] == c]
            class_counter[0, c] = gt_c.shape[0]
            preds_c = preds[preds[:, 4] == c]
            if preds_c.shape[0] > 0:
                match_table = compute_match_table(preds_c, gt_c, self.imgs_counter)
                self.match_table[c] = self.match_table[c].append(match_table)
        self.imgs_counter = self.imgs_counter + 1
        self.class_counter = torch.cat((self.class_counter, class_counter), dim=0)

    def value(self, iou_thresholds=[0.5], recall_thresholds=None, mpolicy="greedy"):
        """ Evaluate Mean Average Precision.
        Arguments:
            iou_thresholds (list of float): IOU thresholds.
            recall_thresholds (torch.tensor or None): specific recall thresholds to the
                                                  computation of average precision.
            mpolicy (str): box matching policy.
                           greedy - greedy matching like VOC PASCAL.
                           soft - soft matching like COCO.
        Returns:
            metric (dict): evaluated metrics."""

        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]

        metric = {}
        aps = torch.zeros((0, self.num_classes), dtype=torch.float32)
        for t in iou_thresholds:
            metric[t] = {}
            aps_t = torch.zeros((1, self.num_classes), dtype=torch.float32)
            for class_id in range(self.num_classes):
                aps_t[0, class_id], precision, recall = self._evaluate_class(
                    class_id, t, recall_thresholds, mpolicy
                )
                metric[t][class_id] = {}
                metric[t][class_id]["ap"] = aps_t[0, class_id]
                metric[t][class_id]["precision"] = precision
                metric[t][class_id]["recall"] = recall
            aps = torch.cat((aps, aps_t), dim=0)
        metric["mAP"] = aps.mean(dim=1).mean(dim=0)
        return metric

    def _evaluate_class(self, class_id, iou_threshold, recall_thresholds, mpolicy="greedy"):
        """ Evaluate class.
        Arguments:
            class_id (int): index of evaluated class.
            iou_threshold (float): iou threshold.
            recall_thresholds (torch.tensor or None): specific recall thresholds to the
                                                  computation of average precision.
            mpolicy (str): box matching policy.
                           greedy - greedy matching like VOC PASCAL.
                           soft - soft matching like COCO.
        Returns:
            average_precision (torch.tensor)
            precision (torch.tensor)
            recall (torch.tensor)
        """
        table = self.match_table[class_id].sort_values(by=['confidence'], ascending=False)
        matched_ind = {}
        nd = len(table)
        tp = torch.zeros(nd, dtype=torch.float64)
        fp = torch.zeros(nd, dtype=torch.float64)
        for d in range(nd):
            img_id, conf, iou, difficult, crowd, order = row_to_vars(table.iloc[d])
            if img_id not in matched_ind:
                matched_ind[img_id] = []
            res, idx = check_box(
                iou,
                difficult,
                crowd,
                order,
                matched_ind[img_id],
                iou_threshold,
                mpolicy
            )
            if res == 'tp':
                tp[d] = 1
                matched_ind[img_id].append(idx)
            elif res == 'fp':
                fp[d] = 1
        precision, recall = compute_precision_recall(tp, fp, self.class_counter[:, class_id].sum())
        if recall_thresholds is None:
            average_precision = compute_average_precision(precision, recall)
        else:
            average_precision = compute_average_precision_with_recall_thresholds(
                precision, recall, recall_thresholds
            )
        return average_precision, precision, recall

    def _init(self):
        """ Initialize internal state."""
        self.imgs_counter = 0
        self.class_counter = torch.zeros((0, self.num_classes), dtype=torch.int32)
        columns = ['img_id', 'confidence', 'iou', 'difficult', 'crowd']
        self.match_table = []
        for i in range(self.num_classes):
            self.match_table.append(pd.DataFrame(columns=columns))
            
def _reverse(t:torch.Tensor):
    idx = [i for i in range(t.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    return t.index_select(0, idx)
    
def sort_by_col(array, idx=1):
    """Sort torch.tensor by column."""
    order = torch.argsort(array[:, idx])[::-1]
    return array[order]

def compute_precision_recall(tp, fp, n_positives):
    """ Compute Preision/Recall.
    Arguments:
        tp (torch.tensor): true positives array.
        fp (torch.tensor): false positives.
        n_positives (int): num positives.
    Returns:
        precision (torch.tensor)
        recall (torch.tensor)
    """
    # TODO ? dim
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)
    recall = tp / max(float(n_positives), 1)
    precision = tp / torch.maximum(tp + fp, torch.tensor(torch.finfo(torch.float64).eps))
    return precision, recall

def compute_average_precision(precision, recall):
    """ Compute Avearage Precision by all points.
    Arguments:
        precision (torch.tensor): precision values.
        recall (torch.tensor): recall values.
    Returns:
        average_precision (torch.tensor)
    """
    precision = torch.cat((torch.tensor([0.]), torch.tensor(precision), torch.tensor([0.])))
    recall = torch.cat((torch.tensor([0.]), torch.tensor(recall), torch.tensor([0.])))
    size = torch.prod(torch.tensor(precision.shape), dim=0)
    for i in range(size - 1, 0, -1):
        precision[i - 1] = torch.maximum(precision[i - 1], precision[i])
    ids = torch.where(recall[1:] != recall[:-1])[0]
    average_precision = torch.sum((recall[ids + 1] - recall[ids]) * precision[ids + 1])
    return average_precision

def compute_average_precision_with_recall_thresholds(precision, recall, recall_thresholds):
    """ Compute Avearage Precision by specific points.
    Arguments:
        precision (torch.tensor): precision values.
        recall (torch.tensor): recall values.
        recall_thresholds (torch.tensor): specific recall thresholds.
    Returns:
        average_precision (torch.tensor)
    """
    average_precision = 0.
    for t in recall_thresholds:
        p = torch.max(precision[recall >= t]) if torch.sum(recall >= t) != 0 else 0
        size = torch.prod(torch.tensor(recall_thresholds.shape), dim=0)
        average_precision = average_precision + p / size
    return average_precision

def compute_iou(pred, gt):
    """ Calculates IoU (Jaccard index) of two sets of bboxes:
            IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)
        Parameters:
            Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
            pred (torch.tensor): predicted bboxes
            gt (torch.tensor): ground truth bboxes
        Return value:
            iou (torch.tensor): intersection over union
    """
    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)

    #_gt = torch.tile(gt, (pred.shape[0], 1))
    _gt = gt.repeat(pred.shape[0], 1)
    _pred = torch.repeat_interleave(pred, gt.shape[0], dim=0)

    ixmin = torch.maximum(_gt[:, 0], _pred[:, 0])
    iymin = torch.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = torch.minimum(_gt[:, 2], _pred[:, 2])
    iymax = torch.minimum(_gt[:, 3], _pred[:, 3])

    width = torch.maximum(ixmax - ixmin + 1., torch.tensor(0))
    height = torch.maximum(iymax - iymin + 1., torch.tensor(0))

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou

def compute_match_table(preds, gt, img_id):
    """ Compute match table.
    Arguments:
        preds (torch.tensor): predicted boxes.
        gt (torch.tensor): ground truth boxes.
        img_id (int): image id
    Returns:
        match_table (pd.DataFrame)
    Input format:
        preds: [xmin, ymin, xmax, ymax, class_id, confidence]
        gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    Output format:
        match_table: [img_id, confidence, iou, difficult, crowd]
    """
    def _tile(arr, nreps, dim=0):
        return torch.repeat_interleave(arr, nreps, dim=dim).reshape(nreps, -1).tolist()

    def _empty_array_2d(size):
        return [[] for i in range(size)]
    
    
    match_table = {}
    match_table["img_id"] = [img_id for i in range(preds.shape[0])]
    match_table["confidence"] = preds[:, 5].tolist()
    if gt.shape[0] > 0:
        match_table["iou"] = compute_iou(preds, gt).tolist()
        match_table["difficult"] = _tile(gt[:, 5], preds.shape[0], dim=0)
        match_table["crowd"] = _tile(gt[:, 6], preds.shape[0], dim=0)
    else:
        match_table["iou"] = _empty_array_2d(preds.shape[0])
        match_table["difficult"] = _empty_array_2d(preds.shape[0])
        match_table["crowd"] = _empty_array_2d(preds.shape[0])
    return pd.DataFrame(match_table, columns=list(match_table.keys()))

def row_to_vars(row):
    """ Convert row of pd.DataFrame to variables.
    Arguments:
        row (pd.DataFrame): row
    Returns:
        img_id (int): image index.
        conf (flaot): confidence of predicted box.
        iou (torch.tensor): iou between predicted box and gt boxes.
        difficult (torch.tensor): difficult of gt boxes.
        crowd (torch.tensor): crowd of gt boxes.
        order (torch.tensor): sorted order of iou's.
    """
    img_id = row["img_id"]
    conf = row["confidence"]
    iou = torch.tensor(row["iou"])
    difficult = torch.tensor(row["difficult"])
    crowd = torch.tensor(row["crowd"])
    order = _reverse(torch.argsort(iou))
    return img_id, conf, iou, difficult, crowd, order

def check_box(iou, difficult, crowd, order, matched_ind, iou_threshold, mpolicy="greedy"):
    """ Check box for tp/fp/ignore.
    Arguments:
        iou (torch.tensor): iou between predicted box and gt boxes.
        difficult (torch.tensor): difficult of gt boxes.
        order (torch.tensor): sorted order of iou's.
        matched_ind (list): matched gt indexes.
        iou_threshold (flaot): iou threshold.
        mpolicy (str): box matching policy.
                       greedy - greedy matching like VOC PASCAL.
                       soft - soft matching like COCO.
    """
    assert mpolicy in ["greedy", "soft"]
    if len(order):
        result = ('fp', -1)
        n_check = 1 if mpolicy == "greedy" else len(order)
        for i in range(n_check):
            idx = order[i]
            if iou[idx] > iou_threshold:
                if not difficult[idx]:
                    if idx not in matched_ind:
                        result = ('tp', idx)
                        break
                    elif crowd[idx]:
                        result = ('ignore', -1)
                        break
                    else:
                        continue
                else:
                    result = ('ignore', -1)
                    break
            else:
                result = ('fp', -1)
                break
    else:
        result = ('fp', -1)
    return result


metrics_dict = {
    'map_2d': MeanAveragePrecision2d
} 


