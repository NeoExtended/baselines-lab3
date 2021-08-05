
def get_schedule(type, **kwargs):
    if type == "linear":
        return LinearSchedule(**kwargs)
    elif type == "piecewise":
        return PiecewiseSchedule(**kwargs)
    else:
        raise NotImplementedError("Only linear and piecewise schedules are available.")


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.
    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)


class LinearSchedule:
    """
    Linear Schedule which gradually interpolates to a final value at a set progress.
    :param initial_value: (float)
        Initial learning rate
    :param final_value: (float)
        Final learning rate
    :param final_point: (float)
        Progress point at which the final value should be reached (1 - start, 0 - end)

    """
    def __init__(self, initial_value, final_value=0.0, final_point=0.0):
        self._initial_value = initial_value
        self._final_value = final_value
        self._final_point = final_point

    def __call__(self, progress):
        # local progress: 1.0 = 100%
        local_progress = max(0.0, (progress - self._final_point) / (1.0 - self._final_point))
        local_progress = 1.0 - local_progress
        return self._initial_value + local_progress * (self._final_value - self._initial_value)


class PiecewiseSchedule:
    """
    Piecewise schedule modified from Stable Baselines but as a callable and with progress instead of steps.
    :param endpoints: ([(float, int)])
        list of pairs `(time, value)` meaning that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """
    def __init__(self, endpoints, interpolation="linear", outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)

        if interpolation == "linear":
            self._interpolation = linear_interpolation
        else:
            raise NotImplementedError("Only linear interpolation is available at the moment.")

        self._endpoints = endpoints
        self._outside_value = outside_value

    def __call__(self, progress):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= progress < right_t:
                alpha = float(progress - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value