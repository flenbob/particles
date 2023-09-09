from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearMixturePackingModel:
    """Predicts final (jammed) packing fraction given sampled diameters using LMPM
    """
    # The LMPM,
    # with modification of L and M as described by Frings, R. M., Kleinhans, M. G., and Vollmer, S.: 
    # "Discriminating between pore-filling load and bed-structure load: a new porosity-based method, 
    # exemplified for the river Rhine", Sedimentology, 55, 1571–1593, 
    # https://doi.org/10.1111/j.1365-3091.2008.00958.x, 2008

    #Equally sized numpy arrays of diameters and corresponding component type_ids
    diameters: np.ndarray
    type_ids: np.ndarray

    #Constants
    ratioOfEntrance: float = 0.154
    n0: float = 0.361

    #Mean diameter, solid volume fractions and porosity parameter component-wise (LMPM input)
    dArray: np.ndarray = np.array([])
    yArray: np.ndarray = np.array([])
    n0Array: np.ndarray = np.array([])

    def __post_init__(self) -> None:
        N = np.max(self.type_ids)

        #Get mean diameter of each component type
        dArray = np.zeros((N, ), dtype=float)
        yArray = np.zeros((N, ), dtype=float)

        volume_tot = 0
        for type_i in range(N):
            #Component mean diameter
            diameters_i = self.diameters[self.type_ids == type_i+1]
            dArray[type_i] = np.mean(diameters_i)

            #Component solid volume
            volume_comp = np.sum(diameters_i**3)
            volume_tot += volume_comp
            yArray[type_i] = volume_comp

        #Solid volume fractions
        yArray = yArray/volume_tot

        #Checks
        if np.isscalar(self.n0):
            n0Array = self.n0*np.ones(len(yArray))
        else:
            n0Array = self.n0
        
        assert len(yArray) == len(dArray), f"Input yArray and dArray do not have matching dimensions ({len(yArray)}) and ({len(dArray)})"
        assert len(dArray) == len(n0Array), f"Input dArray and n0Array do not have matching dimensions ({len(dArray)}) and ({len(n0Array)})"
        
        sum_yArray = np.sum(yArray)
        assert 0.95 <= sum_yArray <= 1.05, f"Sum of solid volume fractions do not add up to 1: {sum_yArray:.3f}"

        #Rescale to match exactly to 1
        if sum_yArray != 1:
            yArray /= sum_yArray

        #Set class variables
        self.dArray = dArray
        self.yArray = yArray
        self.n0Array = n0Array

    def size_ratio(self, d1: float, d2: float) -> float:
        """Returns the size ratio between two given samples

        Args:
            d1 (float): First sample (diameter)
            d2 (float): Second sample (diameter)

        Returns:
            float: Size ratio
        """
        if d1 > d2:
            return d2/d1
        else:
            return d1/d2

    def aij(self, rij: float, A1: float=3.3, A2: float=2.8, A3: float=2.7) -> float:
        """Filling interaction function

        Args:
            rij (float): _description_
            A1 (float, optional): Constant. Defaults to 3.3.
            A2 (float, optional): Constant. Defaults to 2.8.
            A3 (float, optional): Constant. Defaults to 2.7.

        Returns:
            float: Value of evaluated function
        """
        # From Yu, A. B., Zou, R. P., and Standish, N.: "Modifying the Linear Packing Model for Predicting the Porosity 
        # of Nonspherical Particle Mixtures", Industrial & Engineering Chemistry Research, 35, 3730–3741, 
        # https://doi.org/10.1021/ie950616a, 1996
        return (1-rij)**A1 + A2*rij*(1-rij)**A3

    def bij(self, rij: float, B1: float=2.0, B2: float=0.4, B3: float=3.7) -> float:
        """Occupating interaction function

        Args:
            rij (float): _description_
            B1 (float, optional): Constant. Defaults to 2.0.
            B2 (float, optional): Constant. Defaults to 0.4.
            B3 (float, optional): Constant. Defaults to 3.7.

        Returns:
            float: Value of evaluated function
        """
        # From Yu, A. B., Zou, R. P., and Standish, N.: "Modifying the Linear Packing Model for Predicting the Porosity 
        # of Nonspherical Particle Mixtures", Industrial & Engineering Chemistry Research, 35, 3730–3741, 
        # https://doi.org/10.1021/ie950616a, 1996
        return (1-rij)**B1 + B2*rij*(1-rij)**B3
   
    def cij(self, rij: float, rho0: float) -> float:
        """Interacting interaction function

        Args:
            rij (float): _description_
            rho0 (float): _description_

        Returns:
            float: Value of evaluated function
        """
        # From Yu, A. B. and Standish, N.: "Estimation of the porosity of particle mixtures by a linear-mixture packing model",
        # Industrial & engineering chemistry research, 30, 1372–1385, https://doi.org/10.1021/ie00054a045, 1991
        if np.isscalar(rij):
            if rij <= 0.741: 
                return 10.288 * 10**(-1.4566*rho0)*(-1.0002 + 0.1126*rij + 5.8455*rij**2-7.9488*rij**3+3.1222*rij**4)
            else:
                return 0
        else:
            return np.where(rij<=0.741, 10.288 * 10**(-1.4566*rho0)*(-1.0002 + 0.1126*rij + 5.8455*rij**2-7.9488*rij**3+3.1222*rij**4), 0)

    def dij(self, rij: float, rho0: float) -> float:
        """Interacting interaction function

        Args:
            rij (float): _description_
            rho0 (float): _description_

        Returns:
            float: Value of evaluated function
        """
        # From Yu, A. B. and Standish, N.: "Estimation of the porosity of particle mixtures by a linear-mixture packing model",
        # Industrial & engineering chemistry research, 30, 1372–1385, https://doi.org/10.1021/ie00054a045, 1991
        if np.isscalar(rij):
            if rij <= 0.741: 
                return (-1.3092+15.039*rho0-37.453*rho0**2+40.869*rho0**3-17.110*rho0**4)*(-1.0029+0.3589*rij+10.970*rij**2-22.197*rij**3+12.434*rij**4)
            else:
                return 0
        else:
            return np.where(rij<=0.741,
                            (-1.3092+15.039*rho0-37.453*rho0**2+40.869*rho0**3-17.110*rho0**4)*(-1.0029+0.3589*rij+10.970*rij**2-22.197*rij**3+12.434*rij**4),
                            0)

    def predict_packing_fraction(self) -> float:
        """Predicts the packing fraction of given object

        Returns:
            float: Predicted porosity
        """
        #Specific volume array
        VArray = 1/(1-self.n0Array)
        
        #Single component
        N = len(self.yArray)
        if N == 1:
            return (VArray[0]-1)/VArray[0]

        V = 0
        ratioOfEntrance = max(self.ratioOfEntrance, 0)
        for i in np.arange(0, N):
            L = 0 
            M = N-1
            for j in np.arange(0, i):
                if self.dArray[j] * ratioOfEntrance >= self.dArray[i] and self.dArray[i] > self.dArray[j+1] * ratioOfEntrance:
                    L = j+1 # modification from Frings et al 2008, different from Y&S 1991
                    break

            for j in np.arange(i+1,N):
                if self.dArray[j-1] >= self.dArray[i] * ratioOfEntrance and self.dArray[i] * ratioOfEntrance > self.dArray[j]:
                    M = j-1 # modification from Frings et al 2008, different from Y&S 1991
                    break

            #Calculate V0 -> rho0, eps0
            sumY_LToM = sum(self.yArray[L:M+1])

            V0 = 0
            for j in np.arange(L,M+1):
                V0 += self.yArray[j] * VArray[j]
            V0 /= sumY_LToM

            rho0 = 1/V0

            #Compute Vimix
            Vimix1 = 0
            Vimix2 = 0
            for h in np.arange(L,M):
                for l in np.arange(h+1,M+1):
                    rij = self.size_ratio(self.dArray[l], self.dArray[h])
                    Vimix1 += self.cij(rij, rho0) * self.yArray[h] * self.yArray[l]
                    Vimix2 += self.dij(rij, rho0) * self.yArray[h] * self.yArray[l] * (self.yArray[h] - self.yArray[l])
            Vimix1 /= sumY_LToM**2
            Vimix2 /= sumY_LToM**3

            Vimix = V0 + Vimix1 + Vimix2
            
            # Compute VMmixing
            VMmxg = Vimix * sumY_LToM

            VLunmxg = 0
            VSunmxg = 0
            
            for j in np.arange(0, L):
                rij = self.size_ratio(self.dArray[i], self.dArray[j])
                VLunmxg += (Vimix -(Vimix-1)*self.bij(rij))*self.yArray[j] 

            # Compute VSunmixing
            for j in np.arange(M+1,N):
                rij = self.size_ratio(self.dArray[i], self.dArray[j])
                VSunmxg += Vimix*(1-self.aij(rij))* self.yArray[j] 

            # Combine
            Vi = VSunmxg + VMmxg + VLunmxg

            # Find maximum
            V = max(V, Vi)

        if V > 0:
            return 1/V
        else:
            return 1