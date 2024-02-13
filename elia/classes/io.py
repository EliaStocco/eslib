import pickle
from typing import Type, TypeVar

T = TypeVar('T', bound='pickleIO')  # T is a subclass of pickleIO

class pickleIO:
    def to_pickle(self, file):
        """Save the object to a *.pickle file."""
        try:
            with open(file, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error saving to pickle file: {e}")

    @classmethod
    def from_pickle(cls: Type[T], file_path: str) -> T:
        try:
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
            if isinstance(obj, cls):
                return obj
            else:
                raise ValueError(f"Invalid pickle file format. Expected type: {cls.__name__}")
        except FileNotFoundError:
            print(f"Error loading from pickle file: File not found - {file_path}")
        except Exception as e:
            print(f"Error loading from pickle file: {e}")