from functools import wraps
from typing import List

from sample_augment.data.state import StateBundle


# FIRST TRY, REMOVE THIS PROBABLY
def step(state: StateBundle, config_entries: List[str], dependencies: List[str] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # handle dependencies
            if dependencies:
                for dependency in dependencies:
                    # ensure that the dependency has been run
                    if not check_dependency(dependency):
                        raise Exception(f"Dependency {dependency} has not been run")

            state_data = get_state_data_from_somewhere(state)
            config_data = get_config_entries(config_entries)

            return func(state_data, config_data, *args, **kwargs)

        # register the function and its dependencies
        register_step(func.__name__, dependencies)
        return wrapper

    return decorator


