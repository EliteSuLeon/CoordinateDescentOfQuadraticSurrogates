import numpy as np
from sympy import symbols, diff, solve


def coordinate_descent_of_quadratic_surrogate(b, x, descend_iterations, cost_function, x_var):
    """
    Here I implemented the "coordinate descent of quadrdatic surrogates" algorithm. You can find detailed explanation
    of why things are computed in a certain way in the Review.
    The comments in the code are there for you to understand which part belong so which in the Review.

    :param b: transformation matrix
    :param x: preliminary image
    :param cost_function: cost function applied to the image
    :param x_var: variables used in the cost functions
        e.g. x_var = symbols('x=:%d' % number_components)  # Creating x0, ..., xn-1
    :param descend_iterations: amount of coordinate descent iterations until changing the surrogates
    :return: Result image with filters and without filters
    """

    number_pixels, number_components = b.shape  # B is stored by columns
    state_list = np.dot(b, x)  # n_p rows in b and n_p rows in x
    c_min = 0.1  # Setting c_min to be low but not too low
    derivatives = [diff(cost_function, xi) for xi in x_var]  # Only phi_i at the i-th spot

    while not is_minimized(cost_function, state_list, x_var):
        # Setting the derivatives
        q_derivatives = [derivative.subs(zip(x_var, state_list)) for derivative in derivatives]

        # Setting the curvature for each component i
        curvatures = calculate_curvatures(number_components, state_list, derivatives, c_min, x_var)

        # Setting the stepsize denominator for each pixel
        stepsize_denominators = calculate_stepsize_denominators(b, number_pixels, curvatures, number_components)

        # Coordinate descent: Updating x and derivatives
        x, q_derivatives = coordinate_descent(x, q_derivatives, stepsize_denominators, b, curvatures,
                                              descend_iterations,
                                              number_pixels, number_components)

        # Updating state of the image
        state_list = update_state(number_components, state_list, q_derivatives, derivatives, curvatures)

    return state_list, x


def calculate_curvatures(number_components, state_list, derivatives, c_min, x_var):
    # Going through components and calculating the curvature for each one. Then append it to the list.
    curvatures = []
    for i in range(number_components):
        curvature = max(c_min, get_curvatures(derivatives[i], x_var[i]))
        curvatures.append(curvature)
    return curvatures


def calculate_stepsize_denominators(b, number_pixels, curvatures, number_components):
    # Calculating the formula for the step size denominator for each pixel
    stepsize_denominators = []
    for j in range(number_pixels):
        result_j = 0
        # Calculating the sum for each component
        for i in range(number_components):
            b_ij = np.abs(b[j][i])
            result_j += b_ij ** 2 * curvatures[i]
        stepsize_denominators.append(result_j)
    return stepsize_denominators


def coordinate_descent(x, q_derivatives, stepsize_denominators, b, curvatures, descend_iterations, number_pixels,
                       number_components):
    # We will go through all pixels descend_iteration times to minimize the cost function for each pixel
    for m in range(descend_iterations):
        for j in range(number_pixels):
            x_old = x[j]
            sum = 0

            for i in range(number_components):
                sum += b[j][i] * q_derivatives[i]

            x_j_candidate = x_old - (1 / stepsize_denominators[j]) * sum

            if x_j_candidate > 0:
                x[j] = x_j_candidate
            else:
                x[j] = 0

            for i in range(number_components):
                if b[j][i] != 0:
                    q_derivatives[i] = q_derivatives[i] + curvatures[i] * (x[j] - x_old) * b[j][i]
    return x, q_derivatives


def update_state(number_components, state_list, q_derivatives, derivatives, curvatures):
    # Computing the new state for each component
    for i in range(number_components):
        state_list[i] = state_list[i] + (q_derivatives[i] - derivatives[i]) / curvatures[i]
    return state_list


def is_minimized(derivatives, state_list, x_var):
    # This checks whether the gradient is 0. If so, we arrived at the minimum.
    for i in range(len(derivatives)):
        if derivatives[i].subs(zip(x_var[i], state_list[i]) != 0):
            return False
    return True


def get_curvatures(derivative, xi):
    # We use the maximum curvature strategy.
    derivative_2 = diff(derivative, xi)
    candidates = solve(derivative_2, xi)
    return max(candidates)