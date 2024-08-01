import pickle
from typing import Type, TypeVar

T = TypeVar('T', bound='pickleIO')  # T is a subclass of pickleIO

class pickleIO:

    def to_file(self:T,file:str)->None:
        self.to_pickle(file)

    @classmethod
    def from_file(cls: Type[T], file: str) -> T:
        return cls.from_pickle(file)

    def to_pickle(self:T, file:str)->None:
        """Save the object to a *.pickle file."""
        try:
            with open(file, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise ValueError(f"Error saving to pickle file: {e}")

    @classmethod
    def from_pickle(cls: Type[T], file: str) -> T:
        """Load an object from a *.pickle file."""
        try:
            with open(file, 'rb') as ff:
                obj = pickle.load(ff)
            if isinstance(obj, cls):
                return obj
            else:
                return cls(obj)
            # else:
            #     raise ValueError(f"Invalid pickle file format. Expected type: {cls.__name__}")
        except FileNotFoundError:
            print(f"Error loading from pickle file: File not found - {file}")
            return None
        except Exception as e:
            print(f"Error loading from pickle file: {e}")
            return obj

    @staticmethod
    def correct_extension_out(func):
        """
        Decorator to redirect the decorated method to `pickleIO.to_pickle` if the file extension is `.pickle`.

        Attention: keyword-only arguments are allowed.
        """
        def wrapper(self:Type[T],**argv):
            if 'file' in argv and isinstance(argv['file'], str) and argv['file'].endswith('.pickle'):
                return self.to_pickle(file=argv['file'])
            else:
                return func(self,**argv)
        return wrapper
    
    @staticmethod
    def correct_extension_in(func):
        """
        Decorator to redirect the decorated method to `pickleIO.from_pickle` if the file extension is `.pickle`.

        Attention: keyword-only arguments are allowed.        
        """
        def wrapper(cls: Type[T],**argv):
            if 'file' in argv and isinstance(argv['file'], str) and argv['file'].endswith('.pickle'):
                return cls.from_pickle(file=argv['file'])
            else:
                return func(cls,**argv)
        return wrapper