import pickle
import jax


def save_pickled_data(path: str, object):
    object = jax.device_get(object)

    with open(path, "wb") as handle:
        pickle.dump(object, handle)


def load_pickled_data(path: str, device_put: bool = False):
    with open(path, "rb") as handle:
        object = pickle.load(handle)

    if device_put:
        return jax.device_put(object)
    else:
        return object
