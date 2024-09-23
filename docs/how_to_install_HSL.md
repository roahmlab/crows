# How to Install `HSL`
- The `ma27` linear solver will be installed as part of the `HSL` installation process.
- The official installation document can be found [here](https://github.com/coin-or-tools/ThirdParty-HSL). This guide provides a more detailed version of the installation process.

## 1. Clone the `ThirdParty-HSL` Repository
From any directory, run the following commands to clone the `ThirdParty-HSL` repository and navigate into it:

```bash
git clone git@github.com:coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL
```
## 2. Obtain `Coin-HSL` Sources

1. Go to the [STFC Portal](https://licences.stfc.ac.uk/product/coin-hsl-archive).

2. Click `ORDER NOW` for the `HSL Academic Licence`. You will need to create an account if you don’t have one.

3. After logging in, download the appropriate `Coin-HSL` archive from your [downloads section](https://licences.stfc.ac.uk/account/downloads).

4. Place the downloaded archive in the `ThirdParty-HSL/` directory and unpack the tarball:

    ```bash
    gunzip coinhsl-x.y.z.tar.gz
    tar xf coinhsl-x.y.z.tar
    ```

5. Rename the directory `coinhsl-x.y.z` to `coinhsl` or set up a symbolic link:

    ```bash
    mv coinhsl-x.y.z coinhsl
    ```
    **OR**
    ```bash
    ln -s coinhsl-x.y.z coinhsl
    ```

## 3. Install Coin-HSL

You can install `Coin-HSL` either system-wide or in a specific folder.

### System-wide Installation

1. Configure the installation. You can see available options with `./configure --help`.
    ```bash
    ./configure
    ```
2. Build the HSL library with:

    ```bash
    make
    ```

3. Install the HSL library and header files with:

    ```bash
    make install
    ```

    **Note**: You may need to use `sudo` for the above commands (e.g., `sudo ./configure`, `sudo make`, `sudo make install`).

### Installation in a Specific Folder

This method is useful if you are on a shared machine, like a server, or if you do not have `sudo` access.

1. Let's say you want to install it to `$HOME/local`.

2. Ensure that `$HOME/local` exists. If not, create it:

    ```bash
    mkdir -p $HOME/local
    ```

3. Configure the installation with the `--prefix` option:

    ```bash
    ./configure --prefix=$HOME/local
    ```

4. Build and install:

    ```bash
    make
    make install
    ```

5. Update your environment variables by adding the following lines to your `~/.bashrc` and then source it:
- `export PATH=$HOME/local/bin:$PATH`
- `export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH`

    ```bash
    echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

## 4. Relink `libcoinhsl.so` to `libhsl.so`
This step is required to link `HSL` to `Ipopt`.
- **System-wide:**

    ```bash
    sudo ln -s /usr/local/lib/libcoinhsl.so /usr/local/lib/libhsl.so
    ```

- **Specific folder:**

    ```bash
    ln -s $HOME/local/lib/libcoinhsl.so $HOME/local/lib/libhsl.so
    ```

## 5. Validate the Installation

If you have `cyipopt` installed, you can validate the installation with the following Python script:

```python
from cyipopt import minimize_ipopt

objective = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
gradient = lambda x: [2 * (x[0] - 1), 2 * (x[1] - 2.5)]

x0 = [2, 2]
res = minimize_ipopt(objective, x0, jac=gradient, options={"linear_solver": "ma27"})

print(res)
```

- **If `HSL` is installed successfully along with `ma27`, you will see:**
```
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

 message: b'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).'
 success: True
  status: 0
     fun: 0.0
       x: [ 1.000e+00  2.500e+00]
     nit: 1
    info:     status: 0
                   x: [ 1.000e+00  2.500e+00]
                   g: []
             obj_val: 0.0
              mult_g: []
            mult_x_L: [ 0.000e+00  0.000e+00]
            mult_x_U: [ 0.000e+00  0.000e+00]
          status_msg: b'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).'
    nfev: 7
    njev: 3
```
- **Otherwise, you will see:**
```
 message: b'Invalid option encountered.'
 success: False
  status: -12
     fun: 0.0
       x: [ 2.000e+00  2.000e+00]
     nit: 0
    info:     status: -12
                   x: [ 2.000e+00  2.000e+00]
                   g: []
             obj_val: 0.0
              mult_g: []
            mult_x_L: [ 0.000e+00  0.000e+00]
            mult_x_U: [ 0.000e+00  0.000e+00]
          status_msg: b'Invalid option encountered.'
    nfev: 0
    njev: 0
```