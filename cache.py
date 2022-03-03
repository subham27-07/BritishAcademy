import hashlib
import functools
import pickle
from redis import Redis

redis = Redis()

def rediscache(keyspace: str, *additional_args):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arghash = hashlib.sha1()
            arghash.update(pickle.dumps(args) + pickle.dumps(kwargs) + pickle.dumps(additional_args))
            prefix = (keyspace + ":").encode()
            cache_key = prefix + arghash.digest()
            cached_value = redis.get(cache_key)

            if cached_value:
                print("Cache hit", func, args, kwargs)
                return pickle.loads(cached_value)

            else:
                print("Cache miss", func, args, kwargs)
                result = func(*args, **kwargs)
                redis.set(cache_key, pickle.dumps(result))
                return result

        return wrapper
    return decorator