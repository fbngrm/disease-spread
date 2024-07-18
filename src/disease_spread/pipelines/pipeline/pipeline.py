from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data ,
                inputs=["features", "labels"],
                outputs=["city_sj", "city_iq"],
                name="preprocess_data_node",
            ),
        ] 
    )