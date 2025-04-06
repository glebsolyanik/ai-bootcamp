import os
import pickle

from semantic_router.routers import SemanticRouter

from utils.state import State

class Router:
    def __init__(
            self, 
            artifacts_path,
            router_config_path,
            index_router_path
        ) -> None:
        
        self.model = SemanticRouter.from_json(
            os.path.join(artifacts_path, router_config_path)
        )

        with open(os.path.join(artifacts_path, index_router_path), 'rb') as f:
            index = pickle.load(f)

        self.model.index = index

    def route_query(self, state:State):
        result = self.model(state["question"], limit=2)

        res = []
        if isinstance(result, list):
            res = [el.name for el in result]
        else:
            res = [result.name]
        return {"context_source": res}
