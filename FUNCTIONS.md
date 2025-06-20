# Function List

### From `data_utils.py`


## `data_row_exists`

Check if the specified column of `new_row` matches the same column of any row in `contents`.

Parameters:
    new_row (list): A list representing the row to check.
    contents (list of lists): The existing rows to search within, where each row is a list.
    column (int, optional): The index of the column to compare. Defaults to 0.

Returns:
    tuple: (exists, index, matched_row)
        exists (bool): True if a matching value is found in the specified column, else False.
        index (int or None): The index of the matching row in `contents` if found, else None.
        matched_row (list or None): The matched row from `contents` if found, else None.

Notes:
    - Matching is based on the value in the specified column.
    - Assumes `new_row` and each row in `contents` have enough columns to access the specified index.
    - Raises IndexError if `column` is out of range for any row.

## `data_exact_row_exists`

Check if a fully identical row exists in `contents`, with options for case and type sensitivity.

Parameters:
    new_row (list): A list representing the row to check.
    contents (list of lists): The existing rows to search within, where each row is a list.
    case_sensitive (bool, optional): If True, string comparisons are case-sensitive. Defaults to False.
    type_sensitive (bool, optional): If True, type mismatches cause non-match. Defaults to False.

Returns:
    tuple: (exists, matched_indices)
        exists (bool): True if one or more matches are found based on the given sensitivity options.
        matched_indices (list): A list of indices in `contents` where rows match `new_row` based on sensitivity.

Notes:
    - Matching requires the same number of elements in each row.
    - If `case_sensitive` is False, string comparisons are case-insensitive.
    - If `type_sensitive` is False, values with equal string representations are considered equal.

## `data_rows_match`

Compares two rows (lists, tuples, etc.) element-wise to determine if they match.

Parameters:
    row1 (iterable): The first row to compare.
    row2 (iterable): The second row to compare.
    case_sensitive (bool): If True, string comparisons are case-sensitive.
                           If False (default), strings are compared case-insensitively.
    type_sensitive (bool): If True, values are compared with type sensitivity (e.g., 1 != "1").
                           If False (default), values are converted to strings before comparison.

Returns:
    bool: True if the rows match element-wise according to the specified sensitivity flags,
          False otherwise.

## `data_remove_rows`

Remove multiple rows from contents based on a list of indices.

Parameters:
    contents (list of lists): The CSV data as a list of rows.
    indices (list or set of int): Indices of rows to remove.

Returns:
    list: New list with specified rows removed.

Notes:
    - Indices that are out of range are ignored.
    - The order of remaining rows is preserved.

## `data_remove_columns`

Remove or keep specified columns from CSV-like data (list of lists).

Parameters:
    contents (list of lists): CSV data, first row is header if inputmode='header'.
    columns (list): List of column indices (int) or header names (str).
    mode (str): 'remove' to drop listed columns, 'keep' to keep only listed columns.
    inputmode (str): 'index' if columns are indices, 'header' if columns are header names.

Returns:
    list of lists: New contents with columns removed or kept accordingly.

Raises:
    ValueError: If invalid mode, inputmode, duplicates, or invalid column names/indices.

## `data_columns_combine`

Combine multiple 1D column vectors or 2D matrices into a single 2D array (column-wise).

Parameters:
    *args: Each argument can be:
        - A 1D NumPy array or list (treated as a column)
        - A 2D NumPy array or list of lists (multiple columns)
        - None or empty arrays/lists are skipped
    check_rows (bool): If True, all inputs must have the same number of rows (vertical elements)

Returns:
    np.ndarray: Combined 2D array of shape (n_rows, total_columns)

Raises:
    ValueError: If an input is a row vector or if row counts mismatch (when check_rows=True)

## `data_bin`

Bin rows of data based on continuous ranges or discrete values.

Parameters:
    data (np.ndarray): 2D array of shape (n_rows, n_cols).
    mode (str): 'range' for numeric range binning, 'value' for discrete value binning.
    dt (float, optional): Bin width for 'range' mode (required if mode='range').
    bin_column (int, optional): Column index for 'value' mode (required if mode='value').
    returnmode (str): 'indices' to return bin indices array, 'binned' to return list of arrays per bin.

Returns:
    If returnmode='indices':
        bin_indices (np.ndarray): Array of shape (n_rows,) assigning each row to a bin index.
                                  Rows outside bins get -1.
        bin_labels (np.ndarray): 
            - For 'range': 1D array of bin edges, length = number_of_bins + 1
            - For 'value': 1D array of unique discrete values used as bins

    If returnmode='binned':
        binned_data (list of np.ndarray): List with one array per bin containing rows in that bin.
        bin_labels (np.ndarray): same as above

## `dataset_combine`

Merge multiple datasets by time binning and trapezoidal averaging.

Parameters:
    *datasets: Each dataset is a 2D array-like of shape (n_rows, n_columns),
               where column 0 is time (in seconds).
    dt (float): Width of time bins.
    avgMode (str): Averaging mode ('arithmetic' or 'trapezoidal').

Returns:
    tuple: (header: list[str], data: np.ndarray, full_data: list[list])

## `dataset_SplitHeaderFromData`

Splits the header row from a dataset if present.

Parameters:
    dataset (list of list): Raw dataset where the first row may be a header.

Returns:
    tuple: (header, data)
        - header (list or None): The header row if detected, else None.
        - data (list of list): The remaining dataset with only numeric rows.

### From `dimless_utils.py`


## `archimedes`

Calculates the Archimedes number (Ar), a dimensionless number representing 
the ratio of gravitational forces to viscous forces.

Formulas:
    - Kinematic mode:
        Ar = g * L**3 * ((rho - rho_l) / rho_l) / nu**2
    - Dynamic mode:
        Ar = g * L**3 * rho_l * (rho - rho_l) / mu**2

Parameters:
    g : float or array-like
        Gravitational acceleration (m/s²)
    L : float or array-like
        Characteristic length (m)
    rho : float or array-like
        Density of the body (kg/m³)
    rho_l : float or array-like
        Density of the fluid (kg/m³)
    mode : str, optional (default: 'dynamic')
        Calculation mode: 'kinematic' or 'dynamic'
    nu : float or array-like, required for 'kinematic' mode
        Kinematic viscosity (m²/s)
    mu : float or array-like, required for 'dynamic' mode
        Dynamic viscosity (Pa·s)

Returns:
    Ar : float or ndarray
        Archimedes number. Propagates np.nan if any input is np.nan.

Raises:
    ValueError: if required parameters are missing or mode is invalid

## `atwood`

Calculates the Atwood number (A), a dimensionless density ratio used to 
characterize density stratification and hydrodynamic instabilities.

Formula:
    A = (rho1 - rho2) / (rho1 + rho2)

Parameters:
    rho1 : float or array-like
        Density of the heavier fluid (kg/m³)
    rho2 : float or array-like
        Density of the lighter fluid (kg/m³)

Returns:
    A : float or ndarray
        Atwood number (dimensionless). Propagates np.nan if any input is np.nan.

## `bagnold`

Calculates the Bagnold number (Ba), a dimensionless number characterizing 
the ratio of grain-inertia to viscous forces in granular flows.

Modes:
    - 'lambda': 
        Ba = (rho * d² * sqrt(lambda) * γ̇) / μ
    - 'concentration': 
        Computes lambda using:
            lambda = 1 / ((phi0 / phi)^(1/3) - 1)
        Then applies:
            Ba = (rho * d² * sqrt(lambda) * γ̇) / μ

Parameters:
    rho : float or array-like
        Particle density (kg/m³)
    d : float or array-like
        Grain diameter (m)
    gamma_dot : float or array-like
        Shear rate (1/s)
    mode : str, optional (default: 'lambda')
        Calculation mode: 'lambda' or 'concentration'
    mu : float or array-like
        Dynamic viscosity of the fluid (Pa·s), required in all modes
    lambda_ : float or array-like, required for 'lambda' mode
        Linear concentration parameter (dimensionless)
    phi : float or array-like, required for 'concentration' mode
        Solids fraction (volume fraction)
    phi0 : float or array-like, required for 'concentration' mode
        Maximum packing concentration (volume fraction)

Returns:
    Ba : float or ndarray
        Bagnold number (dimensionless). Propagates np.nan if any input is np.nan.

Raises:
    ValueError: if required parameters are missing or mode is invalid

## `bejan`

Calculate the Bejan number based on selected mode.

Modes:
1. Entropy generation ratio
   Be = sgen_deltaT / (sgen_deltaT + sgen_deltap)

2. Brinkman relation
   Be = 1 / (1 + brinkman)

3. Heat transfer mode
   Be = (delta_p * L^2) / (mu * alpha)

4. Mass transfer mode
   Be = (delta_p * L^2) / (mu * D)

5. Fluid mechanics mode
   Be = (delta_p * L^2) / (mu * nu)

6. Hagen-Poiseuille flow
   Be = (32 * Re * L^3) / (d^3)

7. Exergy dissipation (exergy form)
   Be = (1 / (Aw * rho * u)) * (L^2 / nu^2) * delta_X

8. Exergy dissipation (entropy form)
   Be = (1 / (Aw * rho * u)) * (T0 * L^2 / nu^2) * delta_S

Parameters:
    mode : int
        Mode selector (1 to 8).
    sgen_deltaT : float or array-like, optional
        Entropy generation rate by heat transfer (W/K), required for mode 1.
    sgen_deltap : float or array-like, optional
        Entropy generation rate by fluid friction (W/K), required for mode 1.
    brinkman : float or array-like, optional
        Brinkman number (dimensionless), required for mode 2.
    delta_p : float or array-like, optional
        Pressure drop (Pa), required for modes 3,4,5.
    L : float or array-like, optional
        Characteristic length (m), required for modes 3,4,5,6,7,8.
    mu : float or array-like, optional
        Dynamic viscosity (Pa·s), required for modes 3,4,5.
    alpha : float or array-like, optional
        Thermal diffusivity (m²/s), required for mode 3.
    D : float or array-like, optional
        Mass diffusivity (m²/s), required for mode 4.
    nu : float or array-like, optional
        Kinematic viscosity (m²/s), required for modes 5,7,8.
    Re : float or array-like, optional
        Reynolds number (dimensionless), required for mode 6.
    d : float or array-like, optional
        Pipe diameter (m), required for mode 6.
    Aw : float or array-like, optional
        Wet area (m²), required for modes 7,8.
    rho : float or array-like, optional
        Density (kg/m³), required for modes 7,8.
    u : float or array-like, optional
        Velocity (m/s), required for modes 7,8.
    delta_X : float or array-like, optional
        Exergy dissipation rate difference (W), required for mode 7.
    delta_S : float or array-like, optional
        Entropy dissipation rate difference (W/K), required for mode 8.
    T0 : float or array-like, optional
        Reference temperature (K), required for mode 8.

Returns:
    float or ndarray:
        Calculated Bejan number(s). np.nan propagates if present in inputs.

Raises:
    ValueError if required parameters for the selected mode are missing or invalid mode.

## `bingham`

Calculates the Bingham number (Bm), a dimensionless number in rheology representing the ratio
of yield stress to viscous stress.

Bm = (tau_y * L) / (mu * V)

Parameters:
    tau_y : float or array-like
        Yield stress (Pa)
    L : float or array-like
        Characteristic length (m)
    mu : float or array-like
        Dynamic viscosity (Pa·s)
    V : float or array-like
        Characteristic velocity (m/s)

Returns:
    Bm : float or ndarray
        Bingham number, with np.nan propagated.

Raises:
    ValueError: if any required parameter is missing

## `blake`

Calculates the Blake number (B), a dimensionless number used in fluid mechanics
to characterize flow in porous media.

B = (u * rho * Dh) / (mu * (1 - epsilon))

Parameters:
    u : float or array-like
        Flow velocity (m/s)
    rho : float or array-like
        Fluid density (kg/m³)
    Dh : float or array-like
        Hydraulic diameter (m)
    mu : float or array-like
        Dynamic viscosity (Pa·s)
    epsilon : float or array-like
        Void fraction (dimensionless, 0 < epsilon < 1)

Returns:
    B : float or ndarray
        Blake number, with np.nan propagated.

Raises:
    ValueError: if any required parameter is missing or if epsilon values are invalid

## `bond`

Calculate the Bond number (Bo), also known as the Eötvös number, which expresses the ratio
of gravitational forces to surface tension forces.

Modes:
1. Direct calculation:
   Bo = (delta_rho * g * L^2) / gamma

2. Using capillary length:
   Bo = (L / capillary_length)^2
   where capillary_length = sqrt(gamma / (rho * g))

3. Using mass, gravity, surface tension, and contact length:
   Bo = (M * g) / (gamma * contact_length)

Parameters:
- delta_rho: Difference in density between two phases (kg/m^3)
- g: Gravitational acceleration (m/s^2)
- L: Characteristic length (m)
- gamma: Surface tension (N/m)
- capillary_length: Capillary length (m)
- M: Mass of the object (kg)
- contact_length: Contact perimeter length (m)

Returns:
- Bond number (dimensionless)

## `brinkman`

Calculate the Brinkman number (Br), a dimensionless quantity representing the ratio of
viscous dissipation to thermal conduction in fluid flow.

Modes:
1. Full definition:
   Br = (μ * u²) / [κ * (Tw - T0)]

2. Reduced form:
   Br = Pr * Ec

Parameters:
- mode: Calculation mode, either 'full' or 'reduced' (default: 'full')

For mode='full':
- mu: Dynamic viscosity μ (Pa·s)
- u: Flow velocity u (m/s)
- kappa: Thermal conductivity κ (W/m·K)
- Tw: Wall temperature (K)
- T0: Bulk fluid temperature (K)

For mode='reduced':
- Pr: Prandtl number (dimensionless)
- Ec: Eckert number (dimensionless)

Returns:
- Brinkman number (dimensionless), as a float or ndarray with np.nan propagated

## `burger`

Calculate the Burger number (Bu), a dimensionless quantity representing the ratio of
rotational to gravitational effects in geophysical flows.

Modes:
1. Dimensionless form:
   Bu = (Ro / Fr)^2

2. Components form:
   Bu = (sqrt(g * H) / (f * L))^2

Parameters:
- mode: Calculation mode, either 'dimensionless' or 'components' (default: 'dimensionless')

For mode='dimensionless':
- Ro: Rossby number (dimensionless)
- Fr: Froude number (dimensionless)

For mode='components':
- g: Gravitational acceleration (m/s^2)
- H: Characteristic depth or height (m)
- f: Coriolis parameter (1/s)
- L: Characteristic horizontal length (m)

Returns:
- Burger number (dimensionless), as a float or ndarray with np.nan propagated

## `biot`

Calculate Biot number in different modes.

Supported modes:
    - 'standard':      Bi = (h * L) / k
    - 'reciprocal':    Bi = k / (h * L)
    - 'radiative':     Bi = (sigma * T^3 * L) / k
    - 'mass':          Bi = (kc / D) * L

Parameters
----------
mode : str
    One of 'standard', 'reciprocal', 'radiative', or 'mass'.

h : float or array-like, optional
    Convective heat transfer coefficient [W/(m²·K)] (used in 'standard').

k : float or array-like, optional
    Thermal conductivity [W/(m·K)] (used in 'standard', 'reciprocal', 'radiative').

L : float or array-like, optional
    Characteristic length [m] (used in all modes).

sigma : float or array-like, optional
    Stefan-Boltzmann constant [W/(m²·K⁴)] (used in 'radiative').

T : float or array-like, optional
    Temperature [K] (used in 'radiative').

kc : float or array-like, optional
    Convective mass transfer coefficient [m/s] (used in 'mass').

D : float or array-like, optional
    Mass diffusivity [m²/s] (used in 'mass').

Returns
-------
Bi : float or np.ndarray
    Calculated Biot number(s), with NaNs propagated.

Raises
------
ValueError
    If the mode is not recognized or required parameters are missing.

## `brownell_katz`

Calculate the Brownell–Katz number (NBK), a dimensionless quantity that characterizes
the relative effects of viscous and capillary forces in porous media flow.

Formula:
    NBK = (u * mu) / (krw * sigma)

Parameters:
- u: Flow velocity (m/s)
- mu: Dynamic viscosity (Pa·s)
- krw: Relative permeability to water (dimensionless)
- sigma: Interfacial tension or surface tension (N/m)

Returns:
- Brownell–Katz number (dimensionless), as a float or ndarray with np.nan propagated

## `capillary`

Calculate the Capillary number (Ca), a dimensionless quantity representing the ratio
of viscous forces to surface tension forces in multiphase flow.

Formula:
    Ca = (mu * V) / sigma

Parameters:
- mu: Dynamic viscosity of the liquid (Pa·s)
- V: Characteristic velocity (m/s)
- sigma: Surface or interfacial tension (N/m)

Returns:
- Capillary number (dimensionless), as a float or ndarray with np.nan propagated

## `cavitation`

Calculate the Cavitation number (Ca), a dimensionless number representing the ratio of
pressure difference to the dynamic pressure in a flowing fluid.

Formula:
    Ca = (p - pv) / (0.5 * rho * v^2)

Parameters:
- p: Local pressure (Pa)
- pv: Vapor pressure of the fluid (Pa)
- rho: Fluid density (kg/m^3)
- v: Characteristic velocity of the flow (m/s)

Returns:
- Cavitation number (dimensionless), scalar or ndarray with np.nan propagated

## `chandrasekhar`

Calculates the Chandrasekhar number (C) in one of three modes.

Mode 1 (classic):
    C = (B0^2 * d^2) / (mu0 * rho * nu * lambda)

Mode 2 (magnetic diffusivity form):
    C = (B^2 * L^2) / (mu0 * mu * Dm)

Mode 3 (from Hartmann number):
    C = Ha^2

Parameters:
    mode : str, optional (default: 'mode1')
        Calculation mode: 'mode1', 'mode2', or 'mode3'

    Mode 1 parameters:
        B0 : float or array-like
            Characteristic magnetic field (T)
        d : float or array-like
            Characteristic length (m)
        mu0 : float or array-like
            Magnetic permeability (H/m)
        rho : float or array-like
            Fluid density (kg/m^3)
        nu : float or array-like
            Kinematic viscosity (m^2/s)
        lambda_ : float or array-like
            Magnetic diffusivity (m^2/s)

    Mode 2 parameters:
        B : float or array-like
            Magnetic field (T)
        L : float or array-like
            Characteristic length (m)
        mu : float or array-like
            Magnetic permeability (relative or absolute)
        Dm : float or array-like
            Magnetic diffusivity

    Mode 3 parameter:
        Ha : float or array-like
            Hartmann number (dimensionless)

Returns:
    C : float or ndarray
        Chandrasekhar number, with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid

## `chilton_colburn`

Calculates the Chilton–Colburn j-factor (J) using one of five modes.

Modes and Formulas:
    Mode 1: J = Nu / (Re * Pr^(1/3))
    Mode 2: J = Sh / (Re * Sc^(1/3))
    Mode 3: J = (h / (cp * G)) * Pr^(2/3)
    Mode 4: J = (kc / v) * Sc^(2/3)
    Mode 5: J = f / 2

Parameters:
    mode : str, optional (default: 'Mode 1')
        Calculation mode: 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', or 'Mode 5'

    Mode 1 parameters:
        Nu : float or array-like
            Nusselt number
        Re : float or array-like
            Reynolds number
        Pr : float or array-like
            Prandtl number

    Mode 2 parameters:
        Sh : float or array-like
            Sherwood number
        Re : float or array-like
            Reynolds number
        Sc : float or array-like
            Schmidt number

    Mode 3 parameters:
        h : float or array-like
            Convective heat transfer coefficient (W/m²·K)
        cp : float or array-like
            Specific heat capacity (J/kg·K)
        G : float or array-like
            Mass flux (kg/m²·s)
        Pr : float or array-like
            Prandtl number

    Mode 4 parameters:
        kc : float or array-like
            Mass transfer coefficient (m/s)
        v : float or array-like
            Average velocity (m/s)
        Sc : float or array-like
            Schmidt number

    Mode 5 parameters:
        f : float or array-like
            Friction factor (dimensionless)

Returns:
    J : float or ndarray
        Chilton–Colburn j-factor, with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid

## `damkohler`

Calculates the Damköhler number (Da) using one of three modes.

Modes and Formulas:
    Mode 1: Da = k * C0^(n-1) * tau
    Mode 2: Da = (k * Q) / (cp * ΔT)
    Mode 3: Da = k * C0^(n-1) / (kg * a)

Parameters:
    mode : str, optional (default: 'Mode 1')
        Calculation mode: 'Mode 1', 'Mode 2', or 'Mode 3'

    Mode 1 parameters:
        k : float or array-like
            Reaction rate constant
        C0 : float or array-like
            Initial concentration
        n : float or array-like
            Reaction order
        tau : float or array-like
            Mean residence time

    Mode 2 parameters:
        k : float or array-like
            Reaction rate constant
        Q : float or array-like
            Process flow rate (energy or mass)
        cp : float or array-like
            Specific heat capacity
        dT : float or array-like
            Temperature difference (ΔT)

    Mode 3 parameters:
        k : float or array-like
            Reaction rate constant
        C0 : float or array-like
            Initial concentration
        n : float or array-like
            Reaction order
        kg : float or array-like
            Global mass transport coefficient
        a : float or array-like
            Interfacial area

Returns:
    Da : float or ndarray
        Damköhler number, with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid

## `darcy_friction_factor`

Calculates the Darcy friction factor f using one of five modes based on available flow and pipe parameters.

Equations for friction factor f solved explicitly and typical use cases:

Mode 1: Pressure loss per unit length (Δp/L)
    f = (2 * D * Δp/L) / (ρ * v²)
    - Used when you know the pressure gradient along the pipe and want to find friction factor.
    - Common in experiments or simulations where pressure drop is directly measured.
    - Requires fluid density, velocity, hydraulic diameter, and pressure loss.

Mode 2: Head loss per unit length (S = Δh/L)
    f = (2 * g * D * S) / v²
    - Uses head loss (height equivalent of pressure loss) instead of direct pressure.
    - Useful when energy loss is measured as a hydraulic head.
    - Requires velocity, hydraulic diameter, gravitational acceleration, and head loss.

Mode 3: Head loss with volumetric flow rate and pipe diameter
    f = (π² * g * D_c⁵ * S) / (8 * Q²)
    - Suitable when volumetric flow rate is measured rather than velocity.
    - Useful for circular pipes where diameter and flow rate are known.
    - Requires head loss, pipe diameter, volumetric flow rate, and gravitational acceleration.

Mode 4: Wall shear stress τ
    f = (8 * τ) / (ρ * v²)
    - Used when wall shear stress is known or measured directly.
    - Common in detailed fluid mechanics studies or CFD simulations.
    - Requires shear stress, fluid density, and velocity.

Mode 5: Dimensionless roughness Reynolds number R_star
    f = (8 * R_star² * D²) / (Re² * ε²)
    - Applies in rough pipe flow characterization.
    - Connects friction factor to relative roughness and flow regime via Reynolds number.
    - Requires dimensionless roughness Reynolds number, pipe diameter, Reynolds number, and surface roughness.

Parameters
----------
mode : str
    Mode selection (case-insensitive):
    'mode1' - from pressure loss per unit length (Δp/L)
    'mode2' - from head loss per unit length (S = Δh/L)
    'mode3' - from head loss per unit length, volumetric flow rate, and pipe diameter
    'mode4' - from wall shear stress τ
    'mode5' - from dimensionless roughness Reynolds number R_star

delta_p_per_L : float or array-like, required for mode1
    Pressure loss per unit length (Pa/m)

rho : float or array-like, required for modes 1 and 4
    Fluid density (kg/m³)

v : float or array-like, required for modes 1, 2, and 4
    Mean flow velocity (m/s)

D : float or array-like, required for modes 1, 2, and 5
    Hydraulic diameter (m)

S : float or array-like, required for modes 2 and 3
    Head loss per unit length (dimensionless m/m)

g : float or array-like, optional, default 9.81
    Gravitational acceleration (m/s²)

Q : float or array-like, required for mode3
    Volumetric flow rate (m³/s)

D_c : float or array-like, required for mode3
    Pipe diameter (m)

tau : float or array-like, required for mode4
    Wall shear stress (Pa)

R_star : float or array-like, required for mode5
    Dimensionless roughness Reynolds number

Re : float or array-like, required for mode5
    Reynolds number

epsilon : float or array-like, required for mode5
    Surface roughness (m)

Returns
-------
f : float or ndarray
    Darcy friction factor, with np.nan propagated

Raises
------
ValueError
    If required parameters for the selected mode are missing or invalid mode is selected.

## `darcy`

Calculates the Darcy number (Da), which quantifies the relative permeability 
of a porous medium with respect to a characteristic length scale.

Darcy number is defined as:
    Da = K / d^2

Parameters:
    K : float or array-like
        Permeability of the medium (m²)
    d : float or array-like
        Characteristic length, e.g., particle diameter (m)

Returns:
    Da : float or ndarray
        Darcy number, with np.nan propagated.

Notes:
    - Inputs can be scalars or arrays; broadcasting is supported.
    - If any input contains np.nan, the result will propagate np.nan accordingly.

## `dean`

Calculates the Dean number (De) for flow in a curved pipe or channel.

The Dean number is a dimensionless number expressing the ratio of inertial and centrifugal forces 
to viscous forces. It characterizes flow in curved pipes and is important for secondary flow analysis.

Two calculation modes are available:

mode 1 (default):
    De = (rho * D * v / mu) * sqrt(D / (2 * Rc))
    Requires fluid density (rho), diameter (D), axial velocity (v), dynamic viscosity (mu), and radius of curvature (Rc).

mode 2:
    De = Re * sqrt(D / (2 * Rc))
    Requires Reynolds number (Re), diameter (D), and radius of curvature (Rc).

Parameters:
    mode : str, optional (default: "mode 1")
        Calculation mode: "mode 1" or "mode 2"
    rho : float or array-like, required for mode 1
        Fluid density (kg/m³)
    D : float or array-like, required
        Diameter (m)
    v : float or array-like, required for mode 1
        Axial velocity scale (m/s)
    mu : float or array-like, required for mode 1
        Dynamic viscosity (Pa·s)
    Rc : float or array-like, required
        Radius of curvature (m)
    Re : float or array-like, required for mode 2
        Reynolds number (dimensionless)

Returns:
    De : float or ndarray
        Dean number, with np.nan propagated.

Raises:
    ValueError: If required parameters for the chosen mode are missing or if mode is invalid.

## `deborah`

Calculates the Deborah number (De), defined as the ratio of the relaxation time to the process time scale.

Deborah number is a dimensionless number that characterizes the fluidity of materials under specific flow conditions.

Parameters:
    tc : float or array-like
        Relaxation time (s)
    tp : float or array-like
        Process or observation time scale (s)

Returns:
    De : float or ndarray
        Deborah number, with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or invalid.

## `drag_coefficient`

Calculates the drag coefficient (cd) using one of two modes:

Mode "force" (default):
    cd = 2 * Fd / (rho * u^2 * A)
    where Fd is the drag force,
          rho is fluid density,
          u is flow speed relative to fluid,
          A is reference area.

Mode "stress":
    cd = tau / q = 2 * tau / (rho * u^2)
    where tau is local shear stress,
          rho is fluid density,
          u is local flow speed,
          q is local dynamic pressure = 0.5 * rho * u^2.

Parameters:
    mode : str, optional (default "force")
        Calculation mode: "force" or "stress".
    Fd : float or array-like, required for mode "force"
        Drag force (N).
    rho : float or array-like, required for both modes
        Fluid density (kg/m³).
    u : float or array-like, required for both modes
        Flow speed relative to fluid (m/s).
    A : float or array-like, required for mode "force"
        Reference area (m²).
    tau : float or array-like, required for mode "stress"
        Local shear stress (Pa).

Returns:
    cd : float or ndarray
        Drag coefficient with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid.

## `dukhin`

Calculates the Dukhin number (Du) using one of three modes.

Modes:
    mode 1:
        Du = kappa_sigma / (Km * a)
        where
            kappa_sigma : surface conductivity
            Km : bulk fluid conductivity
            a : particle size

    mode 2:
        Du = 2 * (1 + 3m / z^2) * kappa * a * (cosh(z * F * zeta / (2 * R * T)) - 1)
        where
            m : electro-osmosis contribution parameter
            kappa : surface conductivity (different from kappa_sigma)
            a : particle size
            z : ion valency
            F : Faraday constant
            zeta : electrokinetic potential
            R : gas constant
            T : absolute temperature

    mode 3:
        Same as mode 2 but calculates m internally as:
            m = (2 * epsilon0 * epsilonm * R^2 * T^2) / (3 * eta * F^2 * D)
        Requires:
            epsilon0 : vacuum permittivity
            epsilonm : fluid permittivity
            eta : dynamic viscosity
            D : diffusion coefficient
            plus all mode 2 parameters except m

Parameters:
    mode : str, optional (default "mode 1")
        Calculation mode: "mode 1", "mode 2", or "mode 3".
    kappa_sigma : float or array-like
        Surface conductivity (required for mode 1).
    Km : float or array-like
        Bulk conductivity (required for mode 1).
    a : float or array-like
        Particle size (required for all modes).
    m : float or array-like
        Electro-osmosis parameter (required for mode 2).
    kappa : float or array-like
        Surface conductivity (required for mode 2 and 3).
    z : float or array-like
        Ion valency (required for mode 2 and 3).
    F : float or array-like
        Faraday constant (required for mode 2 and 3).
    zeta : float or array-like
        Electrokinetic potential (required for mode 2 and 3).
    R : float or array-like
        Gas constant (required for mode 2 and 3).
    T : float or array-like
        Absolute temperature (required for mode 2 and 3).
    epsilon0 : float or array-like
        Vacuum permittivity (required for mode 3).
    epsilonm : float or array-like
        Fluid permittivity (required for mode 3).
    eta : float or array-like
        Dynamic viscosity (required for mode 3).
    D : float or array-like
        Diffusion coefficient (required for mode 3).

Returns:
    Du : float or ndarray
        Dukhin number with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid.

## `eckert`

Calculates the Eckert number (Ec), which represents the ratio of advective transport 
to heat dissipation potential in a flowing continuum.

Ec = u^2 / (cp * delta_T)

Parameters:
    u : float or array-like
        Local flow velocity (m/s)
    cp : float or array-like
        Specific heat at constant pressure (J/kg·K)
    delta_T : float or array-like
        Temperature difference between wall and local temperature (K)

Returns:
    Ec : float or ndarray
        Eckert number, with np.nan propagated.

Raises:
    ValueError: if any input is None or missing.

## `ekman`

Calculates the Ekman number (Ek), a dimensionless number describing the ratio of viscous 
to Coriolis forces in a rotating fluid system. Four modes are supported:

Modes:
    mode 1:
        Ek = nu / (2 * D^2 * Omega * sin(phi))
        Parameters required: nu, D, Omega, phi (radians)
    mode 2 (Tritton):
        Ek = nu / (Omega * L^2)
        Parameters required: nu, L, Omega
    mode 3 (NRL Plasma Formulary):
        Ek = sqrt(nu / (2 * Omega * L^2))
        Parameters required: nu, L, Omega
    mode 4:
        Ek = sqrt(Ro / Re)
        Parameters required: Ro, Re

Parameters:
    nu : float or array-like
        Kinematic viscosity (m^2/s)
    mode : str, optional (default='mode 1')
        Calculation mode: 'mode 1', 'mode 2', 'mode 3', or 'mode 4'
    D : float or array-like, required for mode 1
        Characteristic length scale (m)
    L : float or array-like, required for modes 2 and 3
        Characteristic length scale (m)
    Omega : float or array-like, required for modes 1, 2, 3
        Angular velocity (rad/s)
    phi : float or array-like, required for mode 1
        Latitude (radians)
    Ro : float or array-like, required for mode 4
        Rossby number
    Re : float or array-like, required for mode 4
        Reynolds number

Returns:
    Ek : float or ndarray
        Ekman number, with np.nan propagated.

Raises:
    ValueError: If required parameters are missing or mode is invalid.

## `ericksen`

Calculates the Ericksen number (Er).

Er = (μ * v * L) / K

where:
    μ : float or array-like
        Dynamic viscosity of the fluid (Pa·s)
    v : float or array-like
        Characteristic velocity scale (m/s)
    L : float or array-like
        Characteristic length scale (m)
    K : float or array-like
        Elasticity force (e.g., elasticity modulus times an area)

Returns:
    Er : float or ndarray
        Ericksen number, with np.nan propagated.

Raises:
    ValueError: if any input is None

## `euler`

Calculates the Euler number (Eu) in two common modes.

Mode 1:
    Eu = (p_u - p_d) / (rho * v^2)

Mode 2:
    Eu = (p_u - p_d) / (0.5 * rho * v^2)

Parameters:
    pu : float or array-like
        Upstream pressure (Pa)
    pd : float or array-like
        Downstream pressure (Pa)
    rho : float or array-like
        Fluid density (kg/m^3)
    v : float or array-like
        Characteristic flow velocity (m/s)
    mode : str, optional (default: "mode 1")
        Calculation mode: "mode 1" or "mode 2"

Returns:
    Eu : float or ndarray
        Euler number, with np.nan propagated.

Raises:
    ValueError: if any required parameter is missing or mode is invalid

## `excess_temperature_coefficient`

Calculates the Excess Temperature Coefficient (Θr).

Θr = (cp * (T - Te)) / (Ue^2 / 2)

Parameters:
    cp : float or array-like
        Specific heat at constant pressure (J/kg·K)
    T : float or array-like
        Local temperature (K)
    Te : float or array-like
        Reference temperature (K)
    Ue : float or array-like
        Characteristic velocity scale (m/s)

Returns:
    Θr : float or ndarray
        Excess Temperature Coefficient, with np.nan propagated.

Raises:
    ValueError: if any required parameter is missing

## `fanning_friction_factor`

Calculate the Fanning friction factor using various correlations depending on flow regime and pipe characteristics.

Parameters
----------
mode : str
    The calculation mode specifying which formula to use:
    Mode 1: Laminar flow in round tubes
        f = 16 / Re
    Mode 2: Laminar flow in square channels
        f = 14.227 / Re
    Mode 3: Turbulent flow in hydraulically smooth pipes (Blasius)
        f = 0.0791 / Re^0.25, valid for 2100 < Re < 1e5
    Mode 4: Turbulent flow with Koo correlation
        f = 0.0014 + 0.125 / Re^0.32, valid for 1e4 < Re < 1e7
    Mode 5: Rough pipes using Haaland equation
        1/sqrt(f) = -3.6 * log10(6.9/Re + (epsilon/D / 3.7)^(10/9))
    Mode 6: Swamee–Jain explicit equation (approximation of Colebrook)
        f = 0.0625 / [log10(epsilon/D / 3.7 + 5.74 / Re^0.9)]^2
    Mode 7: Fully rough turbulent flow (Nikuradse)
        1/sqrt(f) = 2.28 - 4.0 * log10(epsilon/D)
    Mode 8: Implicit Colebrook equation (iterative solution)
        1/sqrt(f) = -4.0 * log10(epsilon/D / 3.7 + 1.255 / (Re * sqrt(f)))

Re : float or np.ndarray
    Reynolds number (dimensionless).
epsilon : float, optional
    Pipe roughness (length units), default is 0.
D : float, optional
    Pipe diameter (length units), default is 1.
f_init : float, optional
    Initial guess for friction factor in iterative mode (mode 8), default is 0.02.
max_iter : int, optional
    Maximum number of iterations for mode 8, default is 100.
tol : float, optional
    Tolerance for convergence in mode 8, default is 1e-6.

Returns
-------
f : float or np.ndarray
    Calculated Fanning friction factor.

## `froude`

Calculates the Froude number in various modes depending on the flow/application context.

The Froude number (Fr) is a dimensionless number expressing the ratio of flow inertia to external force fields,
often gravity.

Modes and formulas:

- classical:
    Fr = u / sqrt(g * L)
    Parameters required: u, L, optional g (default 9.80665)

- displacement:
    Fr = u / sqrt(g * V^(1/3))
    Parameters required: u, V, optional g

- shallow_water:
    Fr = U / sqrt(g * (A / B))
    Parameters required: U, A, B, optional g

- shallow_water_simplified:
    Fr = U / sqrt(g * d)
    Parameters required: U, d, optional g

- extended:
    Fr = u / sqrt(beta * h + s_g * (x_d - x))
    Parameters required: u, beta, h, s_g, x_d, x

- stirred_tank:
    Fr = omega * sqrt(r / g)
    Parameters required: omega, r, optional g

- densimetric:
    Fr = u / sqrt(g_prime * h)
    Parameters required: u, g_prime, h

- walking_velocity:
    Fr = v**2 / (g * l)
    Parameters required: v, l, optional g

- walking_frequency:
    Fr = l * f**2 / g
    Parameters required: l, f, optional g

Parameters
----------
mode : str
    Calculation mode, one of ['classical', 'displacement', 'shallow_water', 'shallow_water_simplified',
                               'extended', 'stirred_tank', 'densimetric', 'walking_velocity', 'walking_frequency']

u : float or array-like, optional
    Flow velocity (m/s) for modes classical, displacement, extended, densimetric

g : float or array-like, optional, default=9.80665
    Gravity acceleration (m/s^2)

L : float or array-like, optional
    Characteristic length (m) for classical mode

V : float or array-like, optional
    Volume (m^3) for displacement mode

U : float or array-like, optional
    Average flow velocity (m/s) for shallow_water modes

A : float or array-like, optional
    Cross-sectional area (m^2) for shallow_water mode

B : float or array-like, optional
    Free-surface width (m) for shallow_water mode

d : float or array-like, optional
    Depth (m) for shallow_water_simplified mode

beta : float or array-like, optional
    Parameter β = g K cos ζ for extended mode

h : float or array-like, optional
    Height (m) for extended and densimetric modes

s_g : float or array-like, optional
    Parameter s_g = g sin ζ for extended mode

x_d : float or array-like, optional
    Downslope distance where flow hits horizontal datum (m) for extended mode

x : float or array-like, optional
    Channel downslope position (m) for extended mode

omega : float or array-like, optional
    Impeller frequency (rad/s or rpm) for stirred_tank mode

r : float or array-like, optional
    Impeller radius (m) for stirred_tank mode

g_prime : float or array-like, optional
    Effective gravity for densimetric mode

v : float or array-like, optional
    Velocity (m/s) for walking_velocity mode

l : float or array-like, optional
    Characteristic length (m) for walking modes

f : float or array-like, optional
    Stride frequency (Hz) for walking_frequency mode

Returns
-------
Fr : float or ndarray
    Calculated Froude number(s), with np.nan propagation.

Raises
------
ValueError
    If required parameters for the selected mode are missing or mode is invalid.

## `grashof`

Calculate Grashof number for heat transfer or mass transfer.

Heat: Gr = g * beta * (Ts - T_inf) L^3 / nu^2
Mass: Gr = g * beta_star * (Ca_s - Ca_a) L^3 / nu^2

Parameters:
-----------
g : float or array-like
    Gravitational acceleration (m/s^2).
L : float or array-like
    Characteristic length (m).
nu : float or array-like
    Kinematic viscosity (m^2/s).
mode : str
    'heat' for heat transfer Grashof number,
    'mass' for mass transfer Grashof number.
beta : float or array-like, required if mode='heat'
    Volumetric thermal expansion coefficient (1/K).
Ts : float or array-like, required if mode='heat'
    Surface temperature (K or °C).
T_inf : float or array-like, required if mode='heat'
    Ambient/bulk temperature (K or °C).
beta_star : float or array-like, required if mode='mass'
    Concentration expansion coefficient (1/concentration units).
Ca_s : float or array-like, required if mode='mass'
    Surface concentration of species a.
Ca_a : float or array-like, required if mode='mass'
    Ambient concentration of species a.

Returns:
--------
Gr : float or np.ndarray
    Calculated Grashof number. NaNs propagated if inputs contain NaN.

Raises:
-------
ValueError:
    If required parameters for the selected mode are not provided.

## `nusselt`

Calculate the Nusselt number in two modes:
  - "average": Nu = h * L / k
  - "local": Nu_x = h_x * x / k

Parameters:
-----------
h : float or array-like
    Convective heat transfer coefficient(s).
    For 'average', scalar or array-like broadcastable with L, k.
    For 'local', array-like.
L : float or array-like
    Characteristic length(s).
k : float or array-like
    Thermal conductivity(ies) of the fluid.
mode : str, optional
    Mode of calculation, either "average" or "local".
    Default is "average".
x : array-like, optional
    Positions along the length for local h_x values.
    Required for "local" mode.

Returns:
--------
Nu : float or np.ndarray
    Calculated Nusselt number(s), with np.nan propagated.

Raises:
-------
ValueError:
    If mode is invalid or if required inputs (like x) are missing for local mode.

## `nusselt_integral`

Calculate average Nusselt number by numerically integrating convective heat
transfer coefficient h over length L.

Nu_avg = ( (1/L) * ∫ h_x dx ) * L / k

Parameters:
-----------
L : float
    Characteristic length (scalar).
h : array-like
    Convective heat transfer coefficients along length (vector).
    NaNs replaced by average of neighbors.
k : float
    Thermal conductivity (scalar).

Returns:
--------
Nu_avg : float
    Average Nusselt number, with np.nan propagated.

## `prandtl`

Calculate Prandtl number.

Pr = (cp * mu) / k

Parameters:
-----------
cp : float or array-like
    Specific heat capacity at constant pressure (J/(kg·K)).
mu : float or array-like
    Dynamic viscosity (Pa·s or N·s/m²).
k : float or array-like
    Thermal conductivity (W/(m·K)).

Returns:
--------
Pr : float or np.ndarray
    Prandtl number, with NaN propagated if inputs contain NaN.

Notes:
------

## `reynolds`

Calculates the Reynolds number in either 'dynamic' or 'kinematic' mode.

Re = rho * u * L / mu = u * L / nu

Parameters:
    u : float or array-like
        Flow velocity (m/s)
    L : float or array-like
        Characteristic length (m)
    mode : str, optional (default: 'dynamic')
        Calculation mode: 'kinematic' or 'dynamic'
    nu : float or array-like, required for 'kinematic' mode
        Kinematic viscosity (m^2/s)
    rho : float or array-like, required for 'dynamic' mode
        Fluid density (kg/m^3)
    mu : float or array-like, required for 'dynamic' mode
        Dynamic viscosity (Pa·s)

Returns:
    Re : float or ndarray
        Reynolds number, with np.nan propagated.

Raises:
    ValueError: if required parameters are missing or mode is invalid

### From `file_utils.py`


## `generate_timestamp`

Produces a timestamp in the form YYYY_MM-DD_hh_mm

Returns:
    timestamp : str

## `folder_create`

Creates the folder (including all intermediate directories) if it doesn't exist.

Parameters:
    path (str): Directory path to create.

Returns:
    None

Raises:
    OSError: If the directory cannot be created due to permissions or invalid path.

## `folder_create_structure`

Recursively creates nested folder structures from a dictionary.

Parameters:
    base_path (str): Base directory under which folders will be created.
    structure (dict): Nested dictionary representing folder structure.
        Keys are folder names, values are subfolder dicts or None.

Returns:
    None

Example:
    structure = {
        'A': {'1': None, '2': None},
        'B': {'1': None},
        'C': {'1': None, '2': None, '3': None}
    }
    folder_create_structure('root_folder', structure)

## `file_check`

Check if a file exists at a specified location.

Parameters:
    filename (str): The name of the file to check (e.g., 'data.csv' or 'report.txt').
    path (str, optional): The directory path where the file should be located.
        If None, defaults to the directory of the current script.

Returns:
    bool: True if the file exists at the resolved location, False otherwise.

## `csv_create`

Create a CSV file with the given filename and header row, if it doesn't already exist.

Parameters:
    filename (str): The name of the CSV file to create (e.g., 'data.csv').
    header (list): A list of column names to write as the header row.
    path (str, optional): The directory path where the file should be created.
        If None, defaults to the directory of the current script.

Returns:
    str: The full path to the CSV file.

## `csv_append_row`

Append a single row of data to an existing CSV file.

Parameters:
    filename (str): The name of the CSV file where the row should be appended.
    row (list): A list of values representing a single row to add to the file.
        The order and number of values should match the existing CSV header.
    path (str, optional): The directory path where the file is located.
        If None, defaults to the directory of the current script.

Returns:
    None

Notes:
    - If the file does not exist, this function will raise a FileNotFoundError.
    - Data is appended at the end of the file without modifying existing contents.
    - It is the caller’s responsibility to ensure the row structure matches the CSV.

## `csv_append_data`

Append multiple rows of data to an existing CSV file.

Parameters:
    filename (str): The name of the CSV file where the data should be appended.
    data (list of list): A list of rows, where each row is a list of values.
        Each row should match the column structure of the existing CSV file.
    path (str, optional): The directory path where the file is located.
        If None, defaults to the directory of the current script.

Returns:
    None

Notes:
    - If the file does not exist, this function will raise a FileNotFoundError.
    - All rows in 'data' are appended in the order they appear in the list.
    - It is the caller’s responsibility to ensure each row has the correct format.

## `csv_open`

Open and read the contents of a CSV file, returning all rows as a list.

Parameters:
    filename (str): The name of the CSV file to open and read.
    path (str, optional): The directory path where the file is located.
        If None, defaults to the directory of the current script.

Returns:
    list: A list of rows from the CSV file, where each row is a list of string values.

Notes:
    - If the file does not exist, this function will raise a FileNotFoundError.
    - The entire contents of the CSV are loaded into memory.
    - Prints the contents to the terminal row by row.

## `markdown_write`

Write the given markdown content to a file (overwrites if exists).

Accepts either a string or a list of strings/tuples to format and write.

Parameters:
    text (str | list): Markdown content to write.
        - If a string, writes it directly.
        - If a list of (name, description) tuples, formats as markdown.
        - If a list of strings, writes each string followed by a newline.
    filename (str): Name of the markdown file (e.g., 'README.md').
    path (str, optional): Directory path where the file will be saved.
        If None, uses the current directory.

Returns:
    None

## `markdown_append`

Append the given markdown text to a file (creates file if it doesn't exist).

Accepts either a string or a list of strings/tuples to format and write.

Parameters:
    text (str | list): Markdown content to append.
        - If a string, writes it directly.
        - If a list of (name, description) tuples, formats as markdown.
        - If a list of strings, writes each string followed by a newline.
    filename (str): Name of the markdown file (e.g., 'README.md').
    path (str, optional): Directory path where the file will be saved.
        If None, uses the current directory.

Returns:
    None

## `QR_generate`

Generates a QR code image, saves it to a specified directory, displays a coordinate plot, 
and returns the binary matrix representing the QR code.

Creates the folder (including all intermediate directories) if it doesn't exist.

Parameters:
    data (str): The text or URL to encode in the QR code.
    filename (str): The name of the output PNG file.
    path (str): Directory path to save the QR code image.
    box_size (int): The size of each box in the QR code.
    border (int): The thickness of the border around the QR code.
    colorSquare (str): The color of the QR code squares (default: "black").
    colorBackground (str): The background color of the QR code (default: "white").
    plotQR (bool): If true, will plot the resulting QR code

Returns:
    np.ndarray: 2D NumPy array where 1 represents a colored square and 0 represents background.

Raises:
    OSError: If the directory cannot be created due to permissions or invalid path.

### From `general_utils.py`


## `validate_inputs`

Validates and converts inputs to NumPy arrays of the specified dtype,
ensuring that np.nan values are preserved if dtype supports them.

Parameters:
    *args: Variable length argument list of inputs to be validated and converted.
    dtype: Desired data type of the output arrays (default: float).
    allow_scalar (bool): If False, scalar inputs will raise a ValueError.
    allow_array (bool): If False, array inputs (ndim > 0) will raise a ValueError.
    check_broadcast (bool): If True, checks that all inputs can be broadcast together.

Returns:
    Each validated input as a separate NumPy array (not packed in a tuple).

Raises:
    TypeError: If any input cannot be converted to the specified dtype.
    ValueError: If scalar or array inputs are disallowed but found,
                if inputs cannot be broadcast together,
                or if nan values are present but dtype is integer.

## `extract_functions_with_docstrings`

Parses a Python file and extracts all top-level function names and their docstrings.

Parameters:
    file_path (str): Path to the Python source file.

Returns:
    list of tuple: A list of (function_name, docstring) tuples.
                   If a function has no docstring, a placeholder message is used.

## `list_py_files`

List all Python files in a directory, excluding '__init__.py'.

Parameters:
    directory (str): Path to the directory to scan.

Returns:
    list of str: List of .py file paths.

### From `heattransfer_utils.py`


## `conduction_radial_steady_numerical`

Solve steady-state radial heat conduction in a cylinder numerically with possible temperature-dependent thermal conductivity.

Parameters
----------
r : tuple of floats
    Radial domain as (r0, R), inner and outer radii.
L : float
    Axial length of the cylinder (used to calculate cylindrical surface area).
bc : tuple
    Boundary conditions at r0 as (T0, T1_input, Q):
    - T0: Temperature at inner radius r0.
    - T1_input: Temperature gradient dT/dr at r0 (can be None if unknown).
    - Q: Heat flux at r0 (can be None if unknown).
    Exactly one of T1_input or Q must be provided, or both must be consistent.
k : float or array-like
    Thermal conductivity. Can be:
    - A constant float or int.
    - An array-like [[T0, k0], [T1, k1], ...] specifying temperature-dependent conductivity,
      which is interpolated internally.
gridpoints : int
    Number of points in radial mesh for numerical solution.
solver : str, optional
    ODE solver method for scipy.integrate.solve_ivp (default "RK45").
tol : float, optional
    Relative tolerance for heat flux conservation check (default 1e-3).

Returns
-------
dict
    Dictionary containing:
    - 'r': radial positions array,
    - 'T': temperature distribution array,
    - 'dT/dr': temperature gradient array,
    - 'Q_error': difference between heat flux at inner and outer boundaries (flux conservation error).

Description
-----------
Solves the nonlinear ODE system describing steady radial heat conduction in a cylinder:

    d/dr [k(T) * A(r) * dT/dr] = 0,

where A(r) = 2 * pi * r * L is the cylindrical surface area.

The equation expands to:

    d²T/dr² + (1/k) * (dk/dT) * (dT/dr)² + (1/r) * dT/dr = 0,

with boundary conditions at the inner radius r0:

    T(r0) = T0,
    and either specified temperature gradient dT/dr(r0) = T1_input
    or specified heat flux Q = -k * A(r0) * dT/dr(r0).

The function supports temperature-dependent thermal conductivity via interpolation,
and integrates the ODE system numerically using solve_ivp.

It verifies heat flux conservation between inner and outer boundaries and warns
if the flux discrepancy exceeds the specified tolerance.

Symmetry condition at r=0 is handled by setting the second derivative of temperature to zero.

Warnings are issued if the ODE solver fails or if boundary fluxes are inconsistent.

This method is suitable for modeling steady heat conduction in cylindrical coordinates
with radial dependence and non-constant thermal conductivity.

## `conduction_radial_analytical`

Analytical solution for radial conduction in a multi-layered cylinder.

Returns
-------
sol_guess : dict
    Dictionary with 'r', 'T', 'Q_dot', 'q_dot', and 'dT_dr'

### From `math_utils.py`


## `average`

Compute column-wise averages over a 2D data array with multiple modes.

Parameters:
    data (array-like): 2D iterable of shape (n_samples, n_features). May contain np.nan, None, or ''.
    mode (str): Averaging mode. Options:
        - 'arithmetic': Simple mean (default)
        - 'weighted': Weighted average (requires `weights`)
        - 'geometric': Geometric mean
        - 'harmonic': Harmonic mean
        - 'median': Median
        - 'mode': Most frequent value
        - 'moving': Moving average (requires `window`)
        - 'trapezoidal': Integral-based average (requires `time`)
    weights (array-like): Optional. Weights for 'weighted' mode.
    window (int): Window size for 'moving' average.
    time (array-like): Time values for 'trapezoidal' mode.

Returns:
    numpy.ndarray: 1D or 2D array of averages per column. Missing columns → None; partial NaN → skip; all-NaN → np.nan.

## `safe_divide`

Safely divide two numbers (scalars, arrays, or combinations thereof),
avoiding division by zero or invalid operations and propagating NaNs
from numerator or denominator.

Parameters:
----------
numerator : float, int, array-like
    The numerator(s) in the division.
denominator : float, int, array-like
    The denominator(s) in the division.
warnings: bool
    When True, prints warnings to the terminal

Returns:
-------
result : float or np.ndarray
    The result of the division. Any division by zero or invalid result
    (e.g., inf, -inf, nan) is replaced with np.nan.
    Also, if numerator or denominator is NaN at any position, result is NaN there.

## `derivative_FDM`

Compute the finite difference derivative of a function with uniform spacing.

Parameters:
    y : np.ndarray
        Vector of function values.
    x : np.ndarray
        Vector of grid points (must be uniformly spaced).
    derivative : int
        Order of the derivative to compute (e.g., 1 for first derivative).
    accuracy : int
        Desired order of accuracy (e.g., 2 for O(h^2)).

Returns:
    dy_dx : np.ndarray
        Vector of the derivative values at each point.
    x_used : np.ndarray
        Possibly trimmed x corresponding to dy_dx.
    error_term : str
        The truncation error term in big-O notation, e.g., "O(h^2)".

## `error_catastrophic_cancellation`

Estimate the relative error in a subtraction due to catastrophic cancellation.

Parameters:
    x : float or np.ndarray
        First approximate value(s).
    y : float or np.ndarray
        Second approximate value(s).
    deltax : float or np.ndarray
        Relative error(s) in x (i.e., delta_x / x).
    deltay : float or np.ndarray
        Relative error(s) in y (i.e., delta_y / y).

Returns:
    rel_error : float or np.ndarray
        Estimated relative error in the computed difference (x - y) due to cancellation,
        given by |x * deltax - y * deltay| / |x - y|.

### From `plot_utils.py`


## `rangeplot`

Generates a high-resolution range plot and saves it as a PNG file.

Parameters:
- data: list of numerical values representing the pie chart slices.
- labels: list of category names corresponding to each slice.
- name: the name used to save the file (no extension, will be saved as '.png').
- color_scheme: string representing the Matplotlib colormap to use (default is "Set3").
  Available color schemes include:
  - "Blues": Light to dark blue.
  - "Oranges": Light to dark orange.
  - "Greens": Light to dark green.
  - "Purples": Light to dark purple.
  - "coolwarm": Blue to red (diverging).
  - "PiYG": Pink to green (diverging).
  - "BrBG": Brown to blue-green (diverging).
  - "tab10": Ten distinct, vibrant colors (ideal for categorical data).
  - "tab20": Twenty distinct, vibrant colors (ideal for larger datasets).
  - "Set3": 12 pastel-style colors (ideal for presentations).
  - "Pastel1" / "Pastel2": Soft and light colors (ideal for subtle distinctions).
  - "Dark2": Darker, high-contrast colors (ideal for better visibility).
- path: the directory where the plot image will be saved (default is the current directory).

## `piechart`

Generates a high-resolution bar chart and saves it as a PNG file.

Parameters:
- data: list of lists, where each inner list contains the values (y-values) for a specific category on the x-axis.
- labels: list of category names corresponding to each bar on the x-axis.
- name: the name used to save the file (no extension, will be saved as '.png').
- color_scheme: string representing the Matplotlib colormap to use (default is "Set3").
  Available color schemes include:
  - "Blues": Light to dark blue.
  - "Oranges": Light to dark orange.
  - "Greens": Light to dark green.
  - "Purples": Light to dark purple.
  - "coolwarm": Blue to red (diverging).
  - "PiYG": Pink to green (diverging).
  - "BrBG": Brown to blue-green (diverging).
  - "tab10": Ten distinct, vibrant colors (ideal for categorical data).
  - "tab20": Twenty distinct, vibrant colors (ideal for larger datasets).
  - "Set3": 12 pastel-style colors (ideal for presentations).
  - "Pastel1" / "Pastel2": Soft and light colors (ideal for subtle distinctions).
  - "Dark2": Darker, high-contrast colors (ideal for better visibility).
- path: the directory where the plot image will be saved (default is the current directory).
- ylabel: label for the y-axis (optional).

### From `thermo_utils.py`


## `mass_leakrate`

Calculate mass leak rate (kg/s) from pressure change in an isochoric system with five modes.

Modes and formulas:
    Mode 1:  m_dot = M * V / (R * T) * dP/dt                                (based on molar mass, moles)
    Mode 2:  m_dot = V * M / (R * T) * dRho/dt                              (based on density change)
    Mode 3:  m_dot = m / P * dP/dt                                          (relative pressure change)
    Mode 4:  m_dot = V / (Rspecific * T) * dP/dt                            (using specific gas constant)
    Mode 5:  m_dot = V * ( dRho/dt - dRho/dP|T dP/dt - dRho/dT|P dT/dt )    (Real gas)

    Ensure that all the derivatives are evaluated at the corresponding timestep.

Parameters:
    mode : str, optional
        "Mode 1", "Mode 2", "Mode 3", "Mode4" or "Mode 5"
    M : float
        Molar mass (kg/mol)
    V : float
        Volume (m^3)
    R : float
        Universal gas constant (J/(mol·K))
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    dPdt : array
        Pressure time derivative (Pa/s)
    dTdt : array
        Temperature time derivative (K/s)
    dRhodt : array
        Density time derivative (kg/m^3/s)
    dRhodP : array
        Density pressure derivative at constant temperature (kg/m^3Pa)
    dRhodT : array
        Density Temperature derivative at constant pressure (kg/m^3K)
    m : float
        Current mass (kg)
    Rspecific : float
        Specific gas constant (J/(kg·K))
    

Returns:
    m_dot : float or ndarray
        Mass leak rate in kg/s.

Raises:
    ValueError: if required parameters are missing or mode is invalid.

