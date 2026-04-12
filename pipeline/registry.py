from pipeline.models.lightgbm_model import LightGBMModel
from pipeline.models.mlp import MLPModel

from pipeline.attribution.shap import SHAPAttributor
from pipeline.attribution.lrp import LRPAttributor
from pipeline.attribution.lime import LIMEAttributor

from pipeline.reduction.umap_reducer import UMAPReducer
from pipeline.reduction.pca_reducer import PCAReducer
from pipeline.reduction.tsne_reducer import TSNEReducer
from pipeline.reduction.pacmap_reducer import PaCMAPReducer

from pipeline.clustering.dbscan_clusterer import DBSCANClusterer
from pipeline.clustering.hdbscan_clusterer import HDBSCANClusterer
from pipeline.clustering.kmeans_clusterer import KMeansClusterer

MODELS = {
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}

ATTRIBUTORS = {
    "shap": SHAPAttributor,
    "lrp": LRPAttributor,
    "lime": LIMEAttributor,
}

REDUCERS = {
    "umap": UMAPReducer,
    "pca": PCAReducer,
    "tsne": TSNEReducer,
    "pacmap": PaCMAPReducer,
}

CLUSTERERS = {
    "dbscan": DBSCANClusterer,
    "hdbscan": HDBSCANClusterer,
    "kmeans": KMeansClusterer,
}
