import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
import sklearn.gaussian_process as gp

np.random.seed(123)

def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2)

def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


def train_test_split(x, y, test_size):
    assert len(x) == len(y)
    n_samples = len(x)
    test_indices = np.sort(
        np.random.choice(
            np.arange(n_samples), int(
                n_samples * test_size), replace=False))
    train_indices = np.ones(n_samples, dtype=bool)
    train_indices[test_indices] = False
    test_indices = ~ train_indices

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var):
    plt.figure(figsize=(8, 4))
    plt.title('Gaussian Process Regressor', fontsize=10)

    plt.plot(data_x, data_y, label='objective')
    plt.plot(
        x_train,
        y_train,
        'o',
        label='train data')

    std = np.sqrt(np.abs(var))

    plt.plot(x_test, mu, label='mean')

    plt.fill_between(
        x_test,
        mu + 2 * std,
        mu - 2 * std,
        alpha=.2,
        label='standard deviation')
    plt.legend(
        loc='lower left',
        fontsize=6)

    plt.show()


def gpr(x_train, y_train, x_test, kernel, theta1, theta2, theta3):
    # average
    mu = []
    # variance
    var = []

    train_length = len(x_train)
    test_length = len(x_test)

    K = np.zeros((train_length, train_length))
    for x_idx in range(train_length):
        for x_prime_idx in range(train_length):
            K[x_idx, x_prime_idx] = kernel(
                x_train[x_idx], x_train[x_prime_idx], theta1, theta2, theta3, x_idx == x_prime_idx)

    yy = np.dot(np.linalg.inv(K), y_train)

    for x_test_idx in range(test_length):
        k = np.zeros((train_length,))
        for x_idx in range(train_length):
            k[x_idx] = kernel(
                x_train[x_idx],
                x_test[x_test_idx],
                theta1, theta2, theta3,
                x_idx == x_test_idx)
        s = kernel(
            x_test[x_test_idx],
            x_test[x_test_idx],
            theta1, theta2, theta3,
            x_test_idx == x_test_idx)
        mu.append(np.dot(k, yy))
        kK_ = np.dot(k, np.linalg.inv(K))
        var.append(s - np.dot(kK_, k.T))
    return np.array(mu), np.array(var)


def kernel(x, x_prime, noise, theta_1, theta_2, theta_3):
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0

    return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta


def optimize(x_train, y_train, bounds, initial_params=np.ones(3), n_iter=1000):
    params = initial_params
    bounds = np.atleast_2d(bounds)

    # log transformation
    log_params = np.log(params)
    log_bounds = np.log(bounds)
    log_scale = log_bounds[:, 1] - log_bounds[:, 0]

    def log_marginal_likelihood(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                K[x_idx, x_prime_idx] = kernel(x_train[x_idx], x_train[x_prime_idx],
                                               params[0], params[1], params[2], x_idx == x_prime_idx)

        y = y_train
        yy = np.dot(np.linalg.inv(K), y_train)
        return - (np.linalg.slogdet(K)[1] + np.dot(y, yy))

    lml_prev = log_marginal_likelihood(params)

    thetas_list = []
    lml_list = []
    for _ in range(n_iter):
        move = 1e-2 * np.random.normal(0, log_scale, size=len(params))

        need_resample = (log_params +
                         move < log_bounds[:, 0]) | (log_params +
                                                     move > log_bounds[:, 1])

        while(np.any(need_resample)):
            move[need_resample] = np.random.normal(0, log_scale, size=len(params))[need_resample]
            need_resample = (log_params +
                             move < log_bounds[:, 0]) | (log_params +
                                                         move > log_bounds[:, 1])

        # proposed distribution
        next_log_params = log_params + move
        next_params = np.exp(next_log_params)
        lml_next = log_marginal_likelihood(next_params)

        r = np.exp(lml_next - lml_prev)

        # metropolis update
        if r > 1 or r > np.random.random():
            params = next_params
            log_params = next_log_params
            lml_prev = lml_next
            thetas_list.append(params)
            lml_list.append(lml_prev)

    return thetas_list[np.argmax(lml_list)]


def main():
    n = 100
    data_x = np.linspace(0, 4 * np.pi, n)
    data_y = objective(data_x)

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.70)
    
    #関数optimizeで得られるハイパーパラメータ
    theta1 = 3.9479622
    theta2 = 21.52594284
    theta3 = 2.17656223

    mu, var = gpr(x_train, y_train, x_test, kernel, theta1, theta2, theta3)
    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var)

    kk = gp.kernels.ConstantKernel() * gp.kernels.RBF() + gp.kernels.WhiteKernel()
    model = gp.GaussianProcessRegressor(kernel=kk)
    model.fit(x_train.reshape(len(x_train), -1), y_train.reshape(len(y_train), -1))

    mean, std = model.predict(x_test.reshape(len(x_test), -1), return_std=True)

    plt.figure(figsize=(8, 4))
    plt.title('Gaussian Process Regressor', fontsize=10)
    plt.plot(data_x, data_y, label='objective')
    plt.plot(x_train, y_train, 'o', label='train data')
    plt.fill_between(x_test,
        mean + 2 * std,
        mean - 2 * std,
        alpha=.2,
        label='standard deviation')
    plt.plot(x_test, mean, label='mean')
    plt.legend(loc='lower left', fontsize=6)
    plt.show()

    print(model.kernel_.get_params())

    theta1 = model.kernel_.get_params()['k1__k1__constant_value']
    theta2 = 2 * model.kernel_.get_params()['k1__k2__length_scale']**2
    theta3 = model.kernel_.get_params()['k2__noise_level']

    mu, var = gpr(x_train, y_train, x_test, kernel, theta1, theta2, theta3)
    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var)

if __name__ == "__main__":
    main()