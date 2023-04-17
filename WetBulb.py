########################## Auto Ingest Imagery #################################
# This script is the .py file for a custom python raster function to calculate Wetbulb values
# from Temperature, Pressure, and Relative Humidity. The wet_bulb_py and SaturatedVapourPressureTable 
# classes are pulled from improver https://github.com/metoppv/improver
# Date: 4/17/2023
# Author: Robert Richard rrichard@esri.com

## Instructions:
# 1) Clone ArcGIS conda env
# 2) Install cfunits and irs 
#   conda install -c conda-forge iris
#   conda install -c conda-forge cfunits
# 3) Place this .py script inside of software directory
#   ArcGIS Pro: C:\Program Files\ArcGIS\Pro\Resources\Raster\Functions\Custom
#   Server: C:\Program Files\ArcGIS\Server\framework\runtime\ArcGIS\Resources\Raster\Functions\Custom
# 4) Build raster function pointing at .py script with Item Group type and provide the Group and Tag Names
# 5) Calculate mosaic dataset fields to populate Group and Tag fields 
# https://pro.arcgis.com/en/pro-app/latest/help/analysis/raster-functions/using-mosaic-dataset-items-in-raster-function-templates.htm

from typing import List, Union
import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList
from numpy import ndarray

# pickling can be used for troubleshooting
#import os
#import pickle
#debug_logs_directory = r'C:\Users\rob10341\OneDrive - Esri\Projects_2023\Met_office\pickle_log'


class wet_bulb_py():
    """
    Attributes
    ----------
    pressure : ndarray
    relative_humidity : ndarray
    temperature : ndarray
    """
    def __init__(self, pressure, relative_humidity, temperature):
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        
        self.precision = 0.005
        self.maximum_iterations = 20
        
        self.SVP_T_MIN = 183.15
        self.SVP_T_MAX = 338.25
        self.SVP_T_INCREMENT = 0.1
        
    def _calculate_latent_heat(self, temperature):
        """
        Calculate a temperature adjusted latent heat of condensation for water
        vapour using the relationship employed by the UM.

        Args:
            temperature:
                Array of air temperatures (K).

        Returns:
            Temperature adjusted latent heat of condensation (J kg-1).
        """
        # RPR Edit
        #: 0 Kelvin in degrees C
        ABSOLUTE_ZERO = -273.15
        # RPR Edit
        #: Latent heat temperature dependence (J K-1 kg-1); from Met Office UM.
        #: Applied to temperatures in Celsius: :math:`LH = 2501 - 2.34 \times 10^3 \times T(celsius)`
        LATENT_HEAT_T_DEPENDENCE = 2.34e3
        #: Latent heat of condensation of water at 0C (J kg-1)
        LH_CONDENSATION_WATER = 2.501e6

        temp_Celsius = temperature + ABSOLUTE_ZERO
        latent_heat = (
            -1.0 * LATENT_HEAT_T_DEPENDENCE * temp_Celsius
            + LH_CONDENSATION_WATER
        )
        return latent_heat
    
    def _svp_table(self) -> ndarray:
        """
        Calculate a saturated vapour pressure (SVP) lookup table.
        The lru_cache decorator caches this table on first call to this function,
        so that the table does not need to be re-calculated if used multiple times.

        A value of SVP for any temperature between T_MIN and T_MAX (inclusive) can be
        obtained by interpolating through the table, as is done in the _svp_from_lookup
        function.

        Returns:
            Array of saturated vapour pressures (Pa).
        """
        svp_data = SaturatedVapourPressureTable(
            t_min=self.SVP_T_MIN, t_max=self.SVP_T_MAX, t_increment=self.SVP_T_INCREMENT
        ).process()
        return svp_data.data
    
    def _svp_from_lookup(self,temperature) -> ndarray:
        """
        Gets value for saturation vapour pressure in a pure water vapour system
        from a pre-calculated lookup table. Interpolates linearly between points in
        the table to the temperatures required.

        Args:
            temperature:
                Array of air temperatures (K).

        Returns:
            Array of saturated vapour pressures (Pa).
        """
        # where temperatures are outside the SVP table range, clip data to
        # within the available range
        t_clipped = np.clip(temperature, self.SVP_T_MIN, self.SVP_T_MAX - self.SVP_T_INCREMENT)

        # interpolate between bracketing values
        table_position = (t_clipped - self.SVP_T_MIN) / self.SVP_T_INCREMENT
        table_index = table_position.astype(int)
        interpolation_factor = table_position - table_index
        svp_table_data = self._svp_table()
        return (1.0 - interpolation_factor) * svp_table_data[
            table_index
        ] + interpolation_factor * svp_table_data[table_index + 1]

    def calculate_svp_in_air(self, temperature: ndarray, pressure: ndarray) -> ndarray:
        """
        Calculates the saturation vapour pressure in air.  Looks up the saturation
        vapour pressure in a pure water vapour system, and pressure-corrects the
        result to obtain the saturation vapour pressure in air.

        Args:
            temperature:
                Array of air temperatures (K).
            pressure:
                Array of pressure (Pa).

        Returns:
            Saturation vapour pressure in air (Pa).

        References:
            Atmosphere-Ocean Dynamics, Adrian E. Gill, International Geophysics
            Series, Vol. 30; Equation A4.7.
        """
        #RPR Edit
        #: 0 Kelvin in degrees C
        ABSOLUTE_ZERO = -273.15
        #RPR Edit
        svp = self._svp_from_lookup(temperature)
        temp_C = temperature + ABSOLUTE_ZERO
        correction = 1.0 + 1.0e-8 * pressure * (4.5 + 6.0e-4 * temp_C * temp_C)
        return svp * correction.astype(np.float32)
    
    def saturated_humidity(self, temperature: ndarray, pressure: ndarray) -> ndarray:
        """
        Calculate specific humidity mixing ratio of saturated air of given temperature and pressure

        Args:
            temperature:
                Air temperature (K)
            pressure:
                Air pressure (Pa)

        Returns:
            Array of specific humidity values (kg kg-1) representing saturated air

        Method from referenced documentation. Note that EARTH_REPSILON is
        simply given as an unnamed constant in the reference (0.62198).

        References:
            ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
        """
        # RPR Edit
        #: Repsilon, ratio of molecular weights of water and dry air (Earth; unitless)
        EARTH_REPSILON = 0.62198
        # RPR Edit
        svp = self.calculate_svp_in_air(temperature, pressure)
        numerator = EARTH_REPSILON * svp
        denominator = np.maximum(svp, pressure) - ((1.0 - EARTH_REPSILON) * svp)
        return (numerator / denominator).astype(temperature.dtype)
    
    def _calculate_specific_heat(self,mixing_ratio: ndarray) -> ndarray:
        """
        Calculate the specific heat capacity for moist air by combining that of
        dry air and water vapour in proportion given by the specific humidity.

        Args:
            mixing_ratio:
                Array of specific humidity (fractional).

        Returns:
            Specific heat capacity of moist air (J kg-1 K-1).
        """
        # RPR Edit
        #: Specific heat capacity of dry air (J K-1 kg-1)
        CP_DRY_AIR = 1005.0
        #: Specific heat capacity of water vapour (J K-1 kg-1)
        CP_WATER_VAPOUR = 1850.0
        # RPR Edit
        specific_heat = (
            -1.0 * mixing_ratio + 1.0
        ) * CP_DRY_AIR + mixing_ratio * CP_WATER_VAPOUR
        return specific_heat

    def _calculate_enthalpy(self,
        mixing_ratio: ndarray,
        specific_heat: ndarray,
        latent_heat: ndarray,
        temperature: ndarray,
    ) -> ndarray:
        """
        Calculate the enthalpy (total energy per unit mass) of air (J kg-1).

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio:
                Array of mixing ratios.
            specific_heat:
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat:
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature:
                Array of air temperatures (K).

        Returns:
           Array of enthalpy values calculated at the same points as the
           input cubes (J kg-1).
        """
        enthalpy = latent_heat * mixing_ratio + specific_heat * temperature
        return enthalpy    
    
    def _calculate_enthalpy_gradient(self,
        mixing_ratio: ndarray,
        specific_heat: ndarray,
        latent_heat: ndarray,
        temperature: ndarray,
    ) -> ndarray:
        """
        Calculate the enthalpy gradient with respect to temperature.

        Method from referenced UM documentation.

        Args:
            mixing_ratio:
                Array of mixing ratios.
            specific_heat:
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat:
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature:
                Array of temperatures (K).

        Returns:
            Array of the enthalpy gradient with respect to temperature.
        """
        # RPR Edit
        #: Specific gas constant for water vapour (J K-1 kg-1)
        R_WATER_VAPOUR = 461.6
        # RPR Edit
        numerator = mixing_ratio * latent_heat * latent_heat
        denominator = R_WATER_VAPOUR * temperature * temperature
        return numerator / denominator + specific_heat    

    def _calculate_wet_bulb_temperature(self, 
        pressure: ndarray, relative_humidity: ndarray, temperature: ndarray
    ) -> ndarray:
        """
        Calculate an array of wet bulb temperatures from inputs in
        the correct units.

        A Newton iterator is used to minimise the gradient of enthalpy
        against temperature. Assumes that the variation of latent heat with
        temperature can be ignored.

        Args:
            pressure:
                Array of air Pressure (Pa).
            relative_humidity:
                Array of relative humidities (1).
            temperature:
                Array of air temperature (K).

        Returns:
            Array of wet bulb temperature (K).

        """
        # Initialise psychrometric variables
        wbt_data_upd = wbt_data = temperature = self.temperature.flatten()
        pressure = self.pressure.flatten()

        latent_heat = self._calculate_latent_heat(wbt_data)
        saturation_mixing_ratio = self.saturated_humidity(wbt_data, pressure)
        mixing_ratio = relative_humidity.flatten() * saturation_mixing_ratio
        specific_heat = self._calculate_specific_heat(mixing_ratio)
        enthalpy = self._calculate_enthalpy(
            mixing_ratio, specific_heat, latent_heat, wbt_data
        )
        del mixing_ratio

        # Iterate to find the wet bulb temperature, using temperature as first
        # guess
        iteration = 0
        to_update = np.arange(temperature.size)
        update_to_update = slice(None)
        while to_update.size and iteration < self.maximum_iterations:

            if iteration > 0:
                wbt_data_upd = wbt_data[to_update]
                pressure = pressure[update_to_update]
                specific_heat = specific_heat[update_to_update]
                latent_heat = latent_heat[update_to_update]
                enthalpy = enthalpy[update_to_update]
                saturation_mixing_ratio = self.saturated_humidity(wbt_data_upd, pressure)

            enthalpy_new = self._calculate_enthalpy(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data_upd
            )
            enthalpy_gradient = self._calculate_enthalpy_gradient(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data_upd
            )
            delta_wbt = (enthalpy - enthalpy_new) / enthalpy_gradient

            # Increment wet bulb temperature at points which have not converged
            update_to_update = np.abs(delta_wbt) > self.precision
            to_update = to_update[update_to_update]
            # RPR Update
            wbt_data[to_update] += delta_wbt[update_to_update].astype('int32')
            # RPR Update
            #wbt_data[to_update] += delta_wbt[update_to_update]

            iteration += 1

        return wbt_data.reshape(temperature.shape)
    
# RPR Edit
class SaturatedVapourPressureTable():
# RPR Edit
#class SaturatedVapourPressureTable(BasePlugin):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    MAX_VALID_TEMPERATURE = 373.0
    MIN_VALID_TEMPERATURE = 173.0

    def __init__(
        self, t_min: float = 183.15, t_max: float = 338.25, t_increment: float = 0.1
    ) -> None:
        """
        Create a table of saturated vapour pressures that can be interpolated
        through to obtain an SVP value for any temperature within the range
        t_min --> (t_max - t_increment).

        The default min/max values create a table that provides SVP values
        covering the temperature range -90C to +65.1C. Note that the last
        bin is not used, so the SVP value corresponding to +65C is the highest
        that will be used.

        Args:
            t_min:
                The minimum temperature for the range, in Kelvin.
            t_max:
                The maximum temperature for the range, in Kelvin.
            t_increment:
                The temperature increment at which to create values for the
                saturated vapour pressure between t_min and t_max.
        """
        self.t_min = t_min
        self.t_max = t_max
        self.t_increment = t_increment

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = (
            "<SaturatedVapourPressureTable: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(self.t_min, self.t_max, self.t_increment)
        )
        return result

    def saturation_vapour_pressure_goff_gratch(self, temperature: ndarray) -> ndarray:
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature:
                Temperature values in Kelvin. Valid from 173K to 373K

        Returns:
            Corresponding values of saturation vapour pressure for a pure
            water vapour system, in hPa.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        constants = {
            1: 10.79574,
            2: 5.028,
            3: 1.50475e-4,
            4: -8.2969,
            5: 0.42873e-3,
            6: 4.76955,
            7: 0.78614,
            8: -9.09685,
            9: 3.56654,
            10: 0.87682,
            11: 0.78614,
        }
        ## RPR EDIT
        #: Triple Point of Water (K)
        TRIPLE_PT_WATER = 273.16
        ## RPR EDIT
        
        triple_pt = TRIPLE_PT_WATER

        # Values for which method is considered valid (see reference).
        # WetBulbTemperature.check_range(temperature.data, 173., 373.)
        if (
            temperature.max() > self.MAX_VALID_TEMPERATURE
            or temperature.min() < self.MIN_VALID_TEMPERATURE
        ):
            msg = "Temperatures out of SVP table range: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))

        svp = temperature.copy()
        for cell in np.nditer(svp, op_flags=["readwrite"]):
            if cell > triple_pt:
                n0 = constants[1] * (1.0 - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (
                    1.0 - np.power(10.0, (constants[4] * (cell / triple_pt - 1.0)))
                )
                n3 = constants[5] * (
                    np.power(10.0, (constants[6] * (1.0 - triple_pt / cell))) - 1.0
                )
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = np.power(10.0, log_es)
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.0)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1.0 - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = np.power(10.0, log_es)

        return svp

    def process(self) -> Cube:
        """
        Create a lookup table of saturation vapour pressure in a pure water
        vapour system for the range of required temperatures.

        Returns:
           A cube of saturated vapour pressure values at temperature
           points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(
            self.t_min, self.t_max + 0.5 * self.t_increment, self.t_increment
        )
        svp_data = self.saturation_vapour_pressure_goff_gratch(temperatures)

        temperature_coord = iris.coords.DimCoord(
            temperatures, "air_temperature", units="K"
        )

        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp = iris.cube.Cube(
            svp_data,
            long_name="saturated_vapour_pressure",
            units="hPa",
            dim_coords_and_dims=[(temperature_coord, 0)],
        )
        svp.convert_units("Pa")
        svp.attributes["minimum_temperature"] = self.t_min
        svp.attributes["maximum_temperature"] = self.t_max
        svp.attributes["temperature_increment"] = self.t_increment

        return svp


class wetBulbRFT():

    def __init__(self):
        self.name = "Wet Bulb"
        self.description = "Converts Pressure, Relative Humidity, and Temperature into Wet Bulb"

    def getParameterInfo(self):
        return [
            {
                'name': 'pressure',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "pressure raster",
                'description': "pressure raster"
            },
            {
                'name': 'relative_humidity',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "relative humidity raster",
                'description': "relative humidity raster"
            },
            {
                'name': 'temperature',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "temperature raster",
                'description': "temperature raster"
            },
        ]

    def getConfiguration(self, **scalars):
        return { 
            #'extractBands': (0,1,2), 
            #'invalidateProperties': 2 | 4 | 8,
            'invalidateProperties': 2 | 4 | 8, #1: XForm, 2: Statistics, 4: Histogram, 8: Key properties
            'inputMask': False,
        }
    
    def updateRasterInfo(self, **kwargs):
        #kwargs['output_info']['bandCount'] = 1
        #kwargs['output_info']['pixelType'] = 'u8'   # ... with int pixel values.
        #kwargs['output_info']['statistics'] = ({'minimum': 1, 'maximum': 2})
        #kwargs['output_info']['histogram'] = (np.array([1,2]))
        #kwargs['output_info']['noData'] = (np.array([0,0]))

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        inst = wet_bulb_py(pixelBlocks['pressure_pixels'], pixelBlocks['relative_humidity_pixels'], pixelBlocks['temperature_pixels'])
        wet_bulb = inst._calculate_wet_bulb_temperature(pixelBlocks['pressure_pixels'], pixelBlocks['relative_humidity_pixels'], pixelBlocks['temperature_pixels'])
        pixelBlocks['output_pixels'] = wet_bulb.reshape(pixelBlocks['temperature_pixels'].shape).astype(props['pixelType'])

        # ########## save a pickle ##########
        # fname = 'pressure.p'
        # pickle_filename = os.path.join(debug_logs_directory, fname)
        # pickle.dump(pixelBlocks['pressure_pixels'], open(pickle_filename,"wb"))
        # ########## pickle saved ##########

        # ########## save a pickle ##########
        # fname = 'relative_humidity.p'
        # pickle_filename = os.path.join(debug_logs_directory, fname)
        # pickle.dump(pixelBlocks['relative_humidity_pixels'], open(pickle_filename,"wb"))
        # ########## pickle saved ##########

        # ########## save a pickle ##########
        # fname = 'temperature.p'
        # pickle_filename = os.path.join(debug_logs_directory, fname)
        # pickle.dump(pixelBlocks['temperature_pixels'], open(pickle_filename,"wb"))
        # ########## pickle saved ##########        

        # ########## save a pickle ##########
        # fname = 'pixelBlocks.p'
        # pickle_filename = os.path.join(debug_logs_directory, fname)
        # pickle.dump(pixelBlocks, open(pickle_filename,"wb"))
        # ########## pickle saved ##########
        return pixelBlocks
    
    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        # if bandIndex == 0:                                    
        #     keyMetadata['bandname'] = 'WetBulb'
        return keyMetadata