from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, split_up, best_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["features", "labels"],
                outputs=["city_sj", "city_iq"],
                name="preprocess_data_node",
            ),
            node(
                func=split_up,
                inputs=["city_sj", "city_iq"],
                outputs=["city_sj_train", "city_sj_test", "city_iq_train", "city_iq_test"],
                name="split_up_node",
            ),

            node(
                func=best_model,
                inputs=["city_sj_train", "city_sj_test", "city_iq_train", "city_iq_test"],
                outputs=["sj_best_model", "iq_best_model"],
                name="best_model_node",
            ),

            node(
                func=predict,
                inputs=["predict_data","sj_best_model", "iq_best_model", "submission_data"],
                outputs="predicted_data",
                name="predict_node",
            ),
        ] 
    )
