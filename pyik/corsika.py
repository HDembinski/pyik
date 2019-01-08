# -*- coding: utf-8 -*-
from __future__ import print_function
from six.moves import range

"""Utility functions to read CORSIKA longitudinal files and headers of ground particle files."""

# transcribed from particle identifiers in CORSIKA manual, only some
# to note Offline first converts to PDG and only has a few names hard-coded
# the packages responsible for that in Offline are not included in these CORSIKA files
# CAVEAT| to view labels correctly need useTex enabled in matplotlibrc
primary_dic = {1: r'$\gamma$', 2: r'$e^{+}$', 3: r'$e^{-}$',
               5: r"$\mu^{+}$", 6: r"$\mu^{-}$",
               7: r"$\pi^{o}$", 8: r"$\pi^{+}$",
               9: r"$\pi^{-}$", 13: "n", 14: "p",
               402: "He", 703: "Li", 904: "Be",
               1105: "B", 1206: "C", 1407: "N",
               1608: "O", 2814: "Si", 5626: "Fe"}


class LongitudinalDataProvider:
    """
    Reads a longitudinal file and provides the data content as numpy arrays.

    The class accepts a filename and automatically reads the file. The arrays are
    accessible as data members of the class. Handles files both written with
    and without the SLANT option.
    """

    def __init__(self, fn):
        import numpy as np
        import math

        self.fn = fn

        lines = file(fn).readlines()

        words = [x for x in lines[0].split() if x]

        n = int(words[3]) - 1
        self.size = n

        if words[4] == "SLANT":
            self.depthType = "slant"
        elif words[4] == "VERTICAL":
            self.depthType = "vertical"
        else:
            raise Exception(
                "Could not determine depth type from first line of longitudinal file")

        self.depth = np.empty(n)

        self.nPhoton = np.empty(n)
        self.nElectronP = np.empty(n)
        self.nElectronM = np.empty(n)
        self.nMuonP = np.empty(n)
        self.nMuonM = np.empty(n)
        self.nHadron = np.empty(n)
        self.nNuclei = np.empty(n)
        self.nCherenkov = np.empty(n)

        columns = (self.depth, self.nPhoton,
                   self.nElectronP, self.nElectronM,
                   self.nMuonP, self.nMuonM,
                   self.nHadron, None, self.nNuclei,
                   self.nCherenkov)

        i = 0
        for line in lines[2:2 + n]:
            for j, x in enumerate(columns):
                if j == 0:
                    if line[:6] == " *****":
                        # correct for overflow assuming equal steps in slant
                        # depth
                        dx = x[1] - x[0]
                        x[i] = x[i - 1] + dx
                    else:
                        x[i] = float(line[:6])
                else:
                    if x is None:
                        continue
                    x[i] = float(line[6 + (j - 1) * 12:6 + j * 12])
            i += 1

        # second to last empty bogus for muons and hadrons (Corsika 6.735 with SLANT option) !
        # extrapolate with power law if value is too low
        def extrapol(entry, x, xp, xpp, n, np, npp):
            log = math.log
            logn = (log(np) - log(npp)) / (xp - xpp) * (x - xp) + log(np)
            if n == 0 or (logn - log(n)) > 0.7:  # = factor of two
                return math.exp(logn)
            else:
                return n

        self.nMuonP[n - 1] = extrapol("MU+", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
                                      self.nMuonP[n - 1], self.nMuonP[n - 2], self.nMuonP[n - 3])
        self.nMuonM[n - 1] = extrapol("MU-", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
                                      self.nMuonM[n - 1], self.nMuonM[n - 2], self.nMuonM[n - 3])
        self.nHadron[n - 1] = extrapol(
            "HADRONS", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
            self.nHadron[n - 1], self.nHadron[n - 2], self.nHadron[n - 3])

        self.eLossPhoton = np.empty(n)
        self.eIonLossEm = np.empty(n)
        self.eCutLossEm = np.empty(n)
        self.eIonLossMuon = np.empty(n)
        self.eCutLossMuon = np.empty(n)
        self.eIonLossHadron = np.empty(n)
        self.eCutLossHadron = np.empty(n)
        self.eLossNeutrino = np.empty(n)

        columns = (self.eLossPhoton,
                   self.eIonLossEm, self.eCutLossEm,
                   self.eIonLossMuon, self.eCutLossMuon,
                   self.eIonLossHadron, self.eCutLossHadron,
                   self.eLossNeutrino, None)

        i = 0
        for line in lines[5 + n:5 + 2 * n]:
            for j, x in enumerate(columns):
                if x is None:
                    continue
                x[i] = float(line[7 + j * 12:7 + (j + 1) * 12])
            i += 1


class SteeringDataProvider:
    """
    Reads a steering card and provides the data content.

    The class accepts a filename and automatically reads the file.
    The data in the steering card is provided in form of attributes of the class.

    Limitations
    -----------
    This class is far from complete. I added only the most interesting things.
    """

    def __init__(self, fn, fixed=False, old=False, lyon=False):
        lines = file(fn).readlines()

        iterator = 0
        for line in lines:
            words = [x for x in line.split() if x]
            if not words:
                continue
            key = words[0]
            if key == "RUNNR":
                self.runnr = int(words[1])
            elif key == "PRMPAR":
                self.primary = int(words[1])
            elif key == "ERANGE":
                self.energyRange = float(
                    words[1]) * 1e9, float(words[2]) * 1e9  # in eV
            elif key == "ESLOPE":
                self.energySlope = float(words[1])
            elif key == "THETAP":
                self.thetaRange = float(words[1]), float(words[2])  # in Deg
            elif key == "PHIP":
                self.phiRange = float(words[1]), float(words[2])  # in Deg

            elif key == "THIN":
                self.thinning = [float(words[1]), float(
                    words[2]), float(words[3])]
            elif key == "ATMOD":
                if old:
                    self.atmod = words[1]
                else:
                    self.atmod = words[2]

            elif (key == "PRIMARY" and words[1] == 'PARTICLE' and words[2] == 'IDENTIFICATION'):
                self.particle = int(words[4])
            elif (key == "PRIMARY" and words[1] == 'ENERGY'):
                self.energy = float(words[5])
            if not fixed:
                if (key == "PRIMARY" and words[1] == 'ANGLES'):
                    if lyon:
                        self.zenith = float(words[6])
                        self.azimuth = float(words[11])
                    else:
                        self.zenith = float(words[5])
                        self.azimuth = float(words[9])
            if fixed:
                if (key == "THETA"):
                    self.zenith = float(words[6])
                if (key == "PHI"):
                    self.azimuth = float(words[6])

            # atmospheric profile, parameters for the top of the atmosphere
            # are added (see CORSIKA manual)
            if old == False:
                if key == "ATMA":
                    self.atma = [float(words[1]), float(
                        words[2]), float(words[3]), float(words[4])]
                    self.atma.append(0.01128292)
                elif key == "ATMB":
                    self.atmb = [float(words[1]), float(
                        words[2]), float(words[3]), float(words[4])]
                    self.atmb.append(1.)
                elif key == "ATMC":
                    self.atmc = [float(words[1]), float(
                        words[2]), float(words[3]), float(words[4])]
                    self.atmc.append(1.e9)
                elif key == "ATMLAY":
                    self.atmlay = [float(words[1]), float(
                        words[2]), float(words[3]), float(words[4])]
                    self.atmlay.insert(0, 0.)

            if old:
                if key == "H" and iterator == 0:
                    lay1 = words
                    iterator += 1
                elif key == "H" and iterator == 1:
                    lay2 = words
                    iterator += 1
                elif key == "H" and iterator == 2:
                    lay3 = words
                    iterator += 1
                elif key == "H" and iterator == 3:
                    lay4 = words
                    iterator += 1
                elif key == "H" and iterator == 4:
                    lay5 = words
                    iterator += 1

            # extend here

        if old:
            try:
                self.atma = [float(lay1[8]), float(lay2[8]),
                             float(lay3[8]), float(lay4[7])]
                self.atma.append(0.01128292)

                self.atmb = [float(lay1[10]), float(lay2[10]),
                             float(lay3[10]), float(lay4[9])]
                self.atmb.append(1.)

                self.atmc = [float(lay1[15][:-1]), float(lay2[15][:-1]),
                             float(lay3[15][:-1]), float(lay4[14][:-1])]
                self.atmc.append(1.e9)
            except:
                self.atma = 0


def IsDataFileValid(filename):
    """ Test whether a CORSIKA particle file is comlete (has a RUN end tag)."""

    import os
    min_size = 26215
    if os.path.exists(filename) and os.stat(filename).st_size > min_size:
        f = file(filename, "rb")
        f.seek(-min_size, 2)
        if 'RUNE' in f.read(min_size):
            return True
    return False


def createShowerInfoLibrary(corsikafilenames, libraryfilename, fixed=False, old=False, lyon=False):
    """
    Read many CORSIKA steering files (*.lst) to create a library that contains the properties of the shower.
    The library is saved to a HDF file.

    CAVEATS
    -------
    relies on parsing of file. if not working for your file, please compare parsing.
    CORSIKA gives energy in GeV, for your analysis may need to correct


    Parameters
    ----------
    corsikafilenames: list of CORSIKA ground particle file names or lst files (in event particle files do not exist)
    libraryfilename: name of shelve file
    fixed: boolean to indicate whether the library had fixed zenith angles or not (changes parsing)

    old: boolean to indicate whether the library used an older CORSIKA version (changes parsing)
    lyon: boolean to indicate whether the library is the Lyon one or not (changes parsing)

    Examples
    --------
    >> from pyik import corsika
    >> from glob import glob
    >> corsika.createShowerInfoLibrary( glob("/data/pQGSJet/DAT00084*lst"), "mylibrary")
    """

    from sys import stdout
    from os import path
    import fnmatch
    from collections import defaultdict
    import pandas as pd

    # ensures that users will submit a list of files
    if not isinstance(corsikafilenames, list):
        print("Program created for multiple inputs.")
        return

    atmos_values = defaultdict(list)
    bad_files = []
    for ifn, fn in enumerate(sorted(corsikafilenames)):
        # ensure only CORSIKA *.lst files checked
        if (not fnmatch.fnmatchcase(fn, "*DAT*") or not fnmatch.fnmatchcase(fn, "*.lst")):
            continue

        if not path.exists(fn):
            print("### warning: file not found", fn)
            continue
        if (not IsDataFileValid(fn) and not fnmatch.fnmatchcase(fn, "*.lst")):
            print(fn, "has no valid run end tag, not reading it.")
            continue

        stdout.write("Scanning file %i/%i: %s    \r" %
                     (ifn + 1, len(corsikafilenames), fn))
        stdout.flush()

        if path.exists(fn):
            lst = SteeringDataProvider(fn, fixed, old, lyon)
            if lst.atma == 0:
                bad_files.append(fn)
            else:
                try:
                    atmos_values["id"].append(lst.runnr)
                    # CORSIKA gives energy in GeV, for your analysis may need to correct
                    atmos_values["energy_mc"].append(lst.energy)
                    atmos_values["primary_id"].append(lst.particle)

                    if lst.particle in primary_dic:
                        atmos_values["primary"].append(
                            primary_dic[lst.particle])
                    else:
                        atmos_values["primary"].append("undefined")

                    atmos_values["zenith_mc"].append(lst.zenith)
                    atmos_values["azimuth_mc"].append(lst.azimuth)

                    month = int(lst.atmod[-2:])
                    atmos_values["atm_key"].append(
                        month)  # could be month or season

                    atm_type = "monthly"
                    if month > 12:
                        atm_type = "seasonal"
                    atmos_values["atm_type"].append(atm_type)

                    atmos_values["atma_0"].append(lst.atma[0])
                    atmos_values["atma_1"].append(lst.atma[1])
                    atmos_values["atma_2"].append(lst.atma[2])
                    atmos_values["atma_3"].append(lst.atma[3])
                    atmos_values["atma_4"].append(lst.atma[4])

                    atmos_values["atmb_0"].append(lst.atmb[0])
                    atmos_values["atmb_1"].append(lst.atmb[1])
                    atmos_values["atmb_2"].append(lst.atmb[2])
                    atmos_values["atmb_3"].append(lst.atmb[3])
                    atmos_values["atmb_4"].append(lst.atmb[4])

                    atmos_values["atmc_0"].append(lst.atmc[0])
                    atmos_values["atmc_1"].append(lst.atmc[1])
                    atmos_values["atmc_2"].append(lst.atmc[2])
                    atmos_values["atmc_3"].append(lst.atmc[3])
                    atmos_values["atmc_4"].append(lst.atmc[4])

                except:
                    print("\nAtmosphere information not present in steering file of {}!".format(fn))
        else:
            print("\nCannot read steering file, information will not be available!")

    print("\nBad files:")
    for bad in bad_files:
        print(bad)

    atm_arr = pd.DataFrame(atmos_values)
    atm_arr.set_index(['id'], drop=False, inplace=True)
    atm_arr.to_hdf(libraryfilename + ".hdf", "atmosphere")
