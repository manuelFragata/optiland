from collections import OrderedDict


class RayCache:

    def __init__(self, cache_size=1000):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def get_rays(self, key):
        return self.cache.get(key)

    def add_rays(self, key, value):
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove the oldest item
        self.cache[key] = value

    def clear_cache(self):
        self.cache = {}
