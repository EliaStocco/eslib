from typing import List
import numpy as np
from icecream import ic

class easyvectorize():

    dtype = None
    array = None

    def __init__(self, dtype=object,array=np.asarray):
        easyvectorize.dtype = dtype
        easyvectorize.array = array

    @staticmethod
    def convert_array(method):
        def wrapper(self, *args, **kwargs):
            # Call the original method
            result = method(self, *args, **kwargs)
            # Convert the result to the target type
            converted_result = easyvectorize.array(result)
            return converted_result
        return wrapper 

    def __call__(self,*argc,**kwarg):

        class vectorized(List[easyvectorize.dtype]):

            def __init__(self,*argc,**kwarg):
                super().__init__(*argc,**kwarg)

            @easyvectorize.convert_array
            def _vectorized_attribute(self, attr):
                # Apply the attribute to each element in self
                return [getattr(element, attr) for element in self]
            
            def _vectorized_method(self, method):
                # Apply the method to each element in self
                @easyvectorize.convert_array
                def _vec_method(*args, **kwargs):
                    return [getattr(element, method)(*args, **kwargs) for element in self]
                return _vec_method

            def __setattr__(self, name, value):
                # Apply __setattr__ to each element in self
                for element in self:
                    setattr(element, name, value)

            def __getattr__(self, name):
                # ic("__getattr__")
                try:
                    # Try to get the attribute from the superclass
                    return super().__getattr__(name)
                except AttributeError:
                    # If not found, raise an AttributeError
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            def __getattribute__(self, name):
                # ic("__getattribute__")
                # Check if the attribute is callable
                try:
                    return super().__getattribute__(name)
                except:
                    if all(callable(getattr(element, name, None)) for element in self):
                        # Apply the method to each element in self
                        return object.__getattribute__(self, '_vectorized_method')(name)
                    else:
                        # Apply __getattribute__ to each element in self
                        return object.__getattribute__(self, '_vectorized_attribute')(name)
                
            def __getitem__(self, key):
                # Overload the __getitem__ method
                value = super().__getitem__(key)
                if type(value) != easyvectorize.dtype:
                    raise ValueError("output type is different from",easyvectorize.dtype)
                else:
                    return value
            
            def __setitem__(self, key, value):
                if type(value) != easyvectorize.dtype:
                    raise ValueError("value type is different from",easyvectorize.dtype)
                else:
                    super().__setitem__(key,value)      

            def get(self,key):
                return self[key]

            def set(self,key,value):
                try:
                    self[key] = value
                except:
                    try:
                        self[key] = easyvectorize.dtype(value)
                    except:
                        raise ValueError("Cannot convert value to",easyvectorize.dtype)

            @easyvectorize.convert_array   
            def apply(self,func:callable,*args, **kwargs):
                return [ func(element,*args, **kwargs) for element in self]
                            
            @easyvectorize.convert_array   
            def call(self,func):
                return [ func(element) for element in self]

        return vectorized(*argc,**kwarg)
    