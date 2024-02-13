import numpy as np


class MovingAverage:
    def __init__(self, window_size):
        """
        Initialize the MovingAverage object.

        Parameters:
        - window_size (int): The size of the moving window.
        """
        self.window_size = window_size
        self.data = np.array([])  # Array to store historical data
        self.moving_averages = np.array([])  # Array to store moving averages
        self.derivatives = np.array([])  # Array to store derivatives

    def update(self, new_data):
        """
        Update the moving average with new data.

        Parameters:
        - new_data (float): New data point to update the moving average.
        """

        self.data = np.append(self.data, new_data)

        if len(self.data) > self.window_size:
            self.data = np.delete(self.data, 0)

        current_ma = np.mean(self.data)
        self.moving_averages = np.append(self.moving_averages, current_ma)

        if len(self.moving_averages) > self.window_size:
            self.moving_averages = np.delete(self.moving_averages, 0)

        self.compute_derivative()

    def compute_derivative(self):
        """
        Compute the derivative of the moving average and update the derivatives array.
        """
        # Compute the derivative of the moving average
        if len(self.moving_averages) >= 2:
            derivative = self.moving_averages[-1] - self.moving_averages[-2]
            self.derivatives = np.append(self.derivatives, derivative)

            # Keep only the last 'window_size' derivatives
            if len(self.derivatives) > self.window_size:
                self.derivatives = np.delete(self.derivatives, 0)

    def get_ma(self):
        """
        Get the current moving average.

        Returns:
        - float: Current moving average.
        """
        return self.moving_averages[-1]

    def get_madt(self):
        """
        Get the time derivative of the moving average.

        Returns:
        - float: Time derivative of the moving average.
        """
        try:
            return self.derivatives[-1]
        except IndexError:
            return np.nan
        
    def state_dict(self):
        state = {
            'window_size': self.window_size,
            'data': self.data.tolist(),
            'moving_averages': self.moving_averages.tolist(),
            'derivatives': self.derivatives.tolist(),
        }
        return state

    def load_state_dict(self,state):
        self.window_size = int(state['window_size'])
        self.data = np.array(state['data'])
        self.moving_averages = np.array(state['moving_averages'])
        self.derivatives = np.array(state['derivatives'])
