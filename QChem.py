#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:06:40 2017

@author: alex
"""
import re
import numpy as np
import logging
from .ParserData import MolecularOrbitals, Amplitudes
from .QCBase import QCMethod, VarNames as V
from .QCBase import var_tag
from .ParserTools import is_square

# create module logger
mLogger = logging.getLogger("CCParser.QChem")

def parse_symmetric_matrix(readlin, n, asmatrix=True):
    """Parse a symmetric matrix printed columnwise 
    
    Parameters
    ----------
    readlin : list
        Readlines list object
    n : int
        Line number of identifier
    asmatrix : bool
        Whether to return a numpy.matrix object or not
    
    Returns
    -------
    numpy.matrix
        Parsed AO matrix as numpy.matrix object if asmatrix=True
    list
        Parsed AO matrix as list of lists if asmatrix=False
    """
    matrix = []
    cols = 0
    index_line = n+1
    first_batch = True
    stop_signals = ["Gap", "=", "eV", "Convergence", "criterion"]
    ncol_ref = len(readlin[index_line].split())
    while True:#loop over blocks
        index_ls = readlin[index_line].split()
        ncol = len(index_ls)
        #if len(index_ls) != ncol \
        if (ncol > ncol_ref or ncol == 0) \
        or any(stop in index_ls for stop in stop_signals):
            break
        if not index_ls[0].isdigit():
            break
        # adding rows scheme -> take line split as is
        j = 0
        if cols > 0:
            first_batch = False
        while True: #loop over lines in one block
            line_split = readlin[index_line+j+1].split()
            if len(line_split) != ncol+1 \
            or any(stop in line_split for stop in stop_signals):
                break
            if not line_split[0].isdigit():
                break
            if first_batch:
                matrix.append([])
            matrix[j] += list(map(float, line_split[1:]))
            j += 1
        index_line += j+1#update index line
        cols += ncol#update total number of columns processed
    if asmatrix: # return np.matrix object
        return np.asmatrix(matrix)
    else: # return list of lists
        return matrix

def parse_AO_matrix(readlin, n, asmatrix=True):
    """Parse a matrix that uses AO descriptors, e.g. MO coefficients 
    
    The matrix is printed columnwise (here hardcoded to max. 6 columns) and has
    AO descriptors on the left side, for instance "25 C 1   pz". The head of
    each column contains the column number (Fortran counting) and the
    eigenvalue, both of which are not parsed.
    
    Parameters
    ----------
    readlin : list
        Readlines list object
    n : int
        Line number of identifier
    asmatrix : bool
        Whether to return a numpy.matrix object or not
    
    Returns
    -------
    numpy.matrix
        Parsed AO matrix as numpy.matrix object if asmatrix=True
    list
        Parsed AO matrix as list of lists if asmatrix=False
    """
    index_line = n+1
    cols = 0
    pattern = r"(\d{1,4}?)\s+(?P<atom>\w{1,3}?\s?(\d+)?)\s+(?!wall)(\w+\s?\d?)\s+(?P<coeffs>(-?\d+\.\d+\s+){1,6})"#hardcoded up to 6 columns
    matrix = []
    while True: # loop over block
        ncol = len(readlin[index_line].split())
        try:
            readlin[index_line].split()[0].isdigit()
        except IndexError:
            break
        if ncol == 0:
            break
        matrix += [[] for _dummy in range(ncol)]
        nbas = 0
        while True: # loop over lines
            match = re.search(pattern, readlin[index_line+nbas+2])
            if match:
                nbas += 1
                tmp = match.group("coeffs").split()
                for k in range(ncol):
                    matrix[cols+k].append(float(tmp[k]))
            else:
                break
        index_line += nbas+2
        cols += ncol
    if asmatrix:
        return np.asmatrix(matrix)
    else:
        return matrix

def parse_inline_vec(line, asarray=True):
    """ Extracts a vector of the format
    '[ 1.000, 2.000, 3.000]' from the current line"""
    pattern = r"[+-]?\d+\.\d*"
    match = re.findall(pattern, line)
    if len(match) > 0:
        match = list(map(float, match))
        return np.asarray(match)

def parse_libwfa_vec(hook_string, data, i):
    """ Extract vector from libwfa block
    which ends with an empty line."""
    j = 1
    vec = []
    while True:
        ls = data[i+j].split()
        if len(ls) == 0:
            break
        elif hook_string in data[i+j]:
            vec = parse_inline_vec(data[i+j])
            break
        j += 1
    return vec

def parse_libwfa_float(hook_string, data, i):
    """ Extracts a float from libwfa block
    which ends with an empty line."""
    j = 1
    dist = 0.0
    while True:
        ls = data[i+j].split()
        if len(ls) == 0:
            break
        elif hook_string in data[i+j]:
            dist = float(ls[-1])
            break
        j += 1
    return dist

class General(QCMethod):
    """Parse general information like basis set, number of atoms, etc. """
    def __init__(self):
        super().__init__()# necessary for every derived class of QCMethod
        # hooks as {function_name : hook_string}
        self.hooks = {'version' : r'Q-Chem\s*(\d+\.\d+), Q-Chem, Inc.,',
                "xyz_coordinates" : "Standard Nuclear Orientation (Angstroms)",
                "electrons" : r"There are \w+(?P<alpha>\d+) alpha and \w+(?P<beta>\d+) beta electrons",
                "nuc_rep" : "Nuclear Repulsion Energy =",
                "n_basis" : r"There are (?P<shells>\d+) shells and (?P<bsf>\d+) basis functions",
                "mulliken": "Ground-State Mulliken Net Atomic Charges",
                "chelpg": "Ground-State ChElPG Net Atomic Charges",
                "has_finished": "Thank you very much for using Q-Chem.",
                "basis_name": "Requested basis set is"}

    @var_tag(V.version)
    def version(self, i, data):
        """ Extract version number of Q-Chem """
        mLogger.info("Q-Chem version number", extra={"Parsed": V.version})
        match = re.search(self.hooks['version'], data[i])
        if match:
            return match.group(1)

    @var_tag(V.xyz_coordinates)
    def xyz_coordinates(self, i, data):
        """ Get XYZ coordinates """
        mLogger.info("XYZ coordinates", extra={"Parsed":V.xyz_coordinates})
        xyz_dat, n = [], 0
        while True:
            if "------------" in data[i+3+n]:
                break
            else:
                xyz_dat.append(data[i+3+n].split()[1:])
                n += 1
        xyz_dat = [[x[0]]+list(map(float,x[1:])) for x in xyz_dat]
        return xyz_dat
    
    @var_tag(V.nuc_repulsion)
    def nuc_rep(self, i, data):
        """ Get nuclear repulsion energy [a.u.] """
        mLogger.info("nuclear repulsion energy",
                     extra={"Parsed":V.nuc_repulsion})
        return float(data[i].split()[-2])

    @var_tag(V.n_bas)
    def n_basis(self, i, data):
        """ Parse number of basis functions """
        match = re.search(self.hooks["n_basis"], data[i])
        if match:
            mLogger.info("number of basis functions",
                         extra={"Parsed":V.n_bas})
            return int(match.group("bsf"))

    @var_tag(V.mulliken)
    def mulliken(self, i, data):
        """ Parse ground-state mulliken charges """
        chg, n = [], 0
        while True:
            if "-------" in data[i+4+n]:
                break
            else:
                chg.append(data[i+4+n].split()[1:])
                n += 1
        chg = [[x[0]]+[float(x[1])] for x in chg]
        mLogger.info("Ground-State Mulliken charges",
                     extra={"Parsed":V.mulliken})
        return chg

    @var_tag(V.chelpg)
    def chelpg(self, i, data):
        """ Parse ground-state ChElPG charges """
        chg, n = [], 0
        while True:
            if "-------" in data[i+4+n]:
                break
            else:
                chg.append(data[i+4+n].split()[1:])
                n += 1
        chg = [[x[0]]+[float(x[1])] for x in chg]
        mLogger.info("Ground-State ChElPG charges", extra={"Parsed":V.chelpg})
        return chg

    @var_tag(V.has_finished)
    def has_finished(self, i, data):
        """ Parse final statement that indicates if Q-Chem finished
        without errors. """
        mLogger.info("whether Q-Chem has finished successfully",
                     extra={"Parsed": V.has_finished})
        return True

    @var_tag(V.basis_name)
    def basis_name(self, i, data):
        """ Parse name of basis set. """
        mLogger.info("basis set name", extra={"Parsed": V.basis_name})
        return data[i].split()[-1]
       

class SCF(QCMethod):
    """ Parse scfman related quantities """
    def __init__(self):
        super().__init__()# necessary for every derived class of QCMethod
        # hooks as {function_name : hook_string}
        self.hooks = {"scf_energy" : "SCF   energy in the final basis set",
                      "mo_energies": "Orbital Energies (a.u.)",
                      "overlap_matrix" : " Overlap Matrix",
                      "orthonorm_matrix" : " Orthonormalization Matrix",
                      "alpha_density_matrix" : " Alpha Density Matrix",
                      "mo_coefficients_r" : "RESTRICTED (RHF) MOLECULAR ORBITAL COEFFICIENTS",
                      "multipole_op" : "Multipole Matrix "}

    @var_tag(V.scf_energy)
    def scf_energy(self, i, data):
        """ Get Hartree-Fock energy [a.u.] from scfman """
        mLogger.info("SCF energy in [a.u.]", extra={"Parsed":V.scf_energy})
        return float(data[i].split()[-1])

    @var_tag(V.mo_energies)
    def mo_energies(self, i, data):
        """ Parse Hartree-Fock molecular orbital energies """
        n = 0
        while True:
            if "------------" in data[i+2+n]:
                n += 1
                break
            else:
                n += 1
        s = "".join(data[i+2:i+2+n])
        occp = r"-- Occupied --([^A-Z]*)-- Virtual --"
        virt = r"-- Virtual --([^A-Z]*)[-]{4,}"
        rem2 = re.search(occp, s, re.MULTILINE)
        rem3 = re.search(virt, s, re.MULTILINE)
        if rem2:  
            a_occ = re.findall(r"-?\d+\.\d+", rem2.group(0), re.M)
        if rem3:
            a_virt = re.findall(r"-?\d+\.\d+", rem3.group(0), re.M)
        alpha = MolecularOrbitals(a_occ, a_virt)
        mLogger.info("molecular orbital energies",
                     extra={"Parsed" : V.mo_energies})
        return alpha

    @var_tag(V.overlap_matrix)
    def overlap_matrix(self, i, data):
        """ Parse overlap matrix S """
        mLogger.info("overlap matrix", extra={"Parsed":V.overlap_matrix})
        return parse_symmetric_matrix(data, i)

    @var_tag(V.orthonorm_matrix)
    def orthonorm_matrix(self, i, data):
        """ Parse orthonormalization matrix X """
        mLogger.info("orthonormalization matrix",
                     extra={"Parsed":V.orthonorm_matrix})
        return parse_symmetric_matrix(data, i)

    @var_tag(V.alpha_dens_mat)
    def alpha_density_matrix(self, i, data):
        """ Parse alpha density matrix P_alpha """
        mLogger.info("SCF alpha density matrix",
                     extra={"Parsed":V.alpha_dens_mat})
        return parse_symmetric_matrix(data, i)

    @var_tag(V.mo_coefficients)
    def mo_coefficients_r(self, i, data):
        """ Parse MO coefficients C for restricted SCF"""
        C = parse_AO_matrix(data, i)
        mLogger.info("{0:} molecular orbital coefficients".format(C.shape[0]),
                     extra={"Parsed":V.mo_coefficients})
        return C

    @var_tag(V.multipole_operator)
    def multipole_op(self, i, data):
        """ Parse Multipole matrix (x,x,x)."""
        M = parse_symmetric_matrix(data, i)
        mLogger.info("Multipole matrix", extra={"Parsed": V.multipole_operator})
        return M

class ADC(QCMethod):
    """ Parse adcman related quantities """
    def __init__(self):
        super().__init__()# necessary for every derived class of QCMethod
        self.hooks = {"scf_energy": "HF Summary",
                      "mp_energy": r"(RI-)?MP\([2-3]\) Summary",
                      "mp_correction": (r"(MP e|E)nergy contribution:\s+"
                                        r"(?P<E>-\d+\.\d+) a.u."),
                      "exc_energies": "Excitation energy:",
                      "osc_strength": "Osc. strength:",
                      "has_converged" : r"Excited state\s+\d+\s+\(.*?\)\s+\[(.*?)\]",
                      "amplitudes": "Important amplitudes:",
                      "total_dipole" : "Total dipole [Debye]",
                      "dipole_moment" : (r"Dip\. moment \[a\.u\.\]:\s+\[\s+(?P<x>-?\d+\.\d+),"
                      r"\s+(?P<y>-?\d+\.\d+),\s+(?P<z>-?\d+\.\d+)\]"),
                      "mulliken_adc": "Mulliken Population Analysis",
                      "diff_detach_mean": "Exciton analysis of the difference density matrix",
                      "diff_attach_mean": "Exciton analysis of the difference density matrix",
                      "diff_da_dist": "Exciton analysis of the difference density matrix",
                      "diff_detach_size": "Exciton analysis of the difference density matrix",
                      "diff_attach_size": "Exciton analysis of the difference density matrix",
                      "diff_detach_mom": "Exciton analysis of the difference density matrix",
                      "diff_attach_mom": "Exciton analysis of the difference density matrix",
                      "trans_hole_mean": "Exciton analysis of the transition density matrix",
                      "trans_elec_mean": "Exciton analysis of the transition density matrix",
                      "trans_eh_dist": "Exciton analysis of the transition density matrix",
                      "trans_hole_size": "Exciton analysis of the transition density matrix",
                      "trans_elec_size": "Exciton analysis of the transition density matrix",
                      "trans_eh_sep": "Exciton analysis of the transition density matrix",
                      "trans_hole_mom": "Exciton analysis of the transition density matrix",
                      "trans_elec_mom": "Exciton analysis of the transition density matrix",
                      "trans_eh_sep_mom": "Exciton analysis of the transition density matrix",
                      "trans_eh_cov": "Exciton analysis of the transition density matrix",
                      "trans_eh_corr": "Exciton analysis of the transition density matrix"}

    @var_tag(V.exc_energy_rel)
    def exc_energies(self, i, data):
        """ Parse excitation energies [eV] """
        mLogger.info("relative ADC(x) excitation energy/-ies [eV]",
                     extra={"Parsed":V.exc_energy_rel})
        return float(data[i].split()[-2])

    @var_tag(V.osc_str)
    def osc_strength(self, i, data):
        """ Parse oscillator strengths """
        mLogger.info("ADC oscillator strength/s",
                     extra={"Parsed":V.osc_str})
        return float(data[i].split()[-1])

    @var_tag(V.scf_energy)
    def scf_energy(self, i, data):
        """ Parse SCF energy [a.u.] from adcman """
        mLogger.info("SCF energy in [a.u.] (adcman)",
                     extra={"Parsed":V.scf_energy})
        return float(data[i+2].split()[-2])

    @var_tag(V.mp_energy)
    def mp_energy(self, i, data):
        """ Parse MP(x) reference state energy [a.u.] from adcman """
        match = re.search(self.hooks["mp_energy"], data[i])
        if match:
            mLogger.info("MP(x) energy in [a.u.] (adcman)",
                         extra={"Parsed":V.mp_energy})
            return float(data[i+3].split()[2])

    @var_tag(V.mp_correction)
    def mp_correction(self, i, data):
        """Parse MP(x) energy contribution in [a.u.] from adcman."""
        match = re.search(self.hooks["mp_correction"], data[i])
        if match:
            mLogger.info("MP(x) correction in [a.u.] (adcman)",
                             extra={"Parsed": V.mp_correction})
            return float(match.group("E"))

    @var_tag(V.has_converged)
    def has_converged(self, i, data):
        """ Parse if state has converged. """
        match = re.search(self.hooks["has_converged"], data[i])
        if match:
            mLogger.info("if states converged (adcman)",
                         extra={"Parsed":V.has_converged})
            return match.group(1) == "converged"

    @var_tag(V.amplitudes)
    def amplitudes(self, i, data):
        """ Parse occ -> virt amplitudes """
        try:
            # regex has the awesome captures feature
            import regex
            expr = r"(?:(?P<orb>\d+) [^0-9]+)+ (?P<ampl>-?\d+\.\d+)" # FIXME: [^0-9] might be a problem if symmetry is on
            srch = regex.search
            have_regex = True
        except ImportError:
             # explicit expression for orbital transitions
             expr = r"(\d+) [^0-9]+ ((\d+) [^0-9]+)? (\d+) [^0-9]+ ((\d+) [^0-9]+)? (-?\d+\.\d+)"
             srch = re.search
             have_regex = False
    
        idx = i+3
        amplist = []# Format: [[occ_i, occ_j,..., virt_a, virt_b,..., ampl], ...]
        while "--------" not in data[idx]:
            match = srch(expr, data[idx])
            if match:
                if have_regex:
                    orb = list(map(int, match.captures("orb")))
                    amp = list(map(float, match.captures("ampl")))
                    amplist.append(orb + amp)
                else:
                    amplist.append([int(match.group(x)) for x in [1,3,4,6] if (match.group(x) != None)]+[float(match.group(7))])
            idx += 1
        mLogger.info("ADC amplitudes", extra={"Parsed":V.amplitudes})
        return Amplitudes.from_list(amplist, factor=2.0)

    @var_tag(V.total_dipole)
    def total_dipole(self, i, data):
        """ Parse total dipole moment [Debye] for HF, MP2 and ADC states
        
        Returns
        -------
        float
            The total dipole moment [Debye]"""
        mLogger.info("Total dipole moment [Debye]",
                     extra={"Parsed":V.total_dipole})
        return float(data[i].split()[-1])

    @var_tag(V.dipole_moment)
    def dipole_moment(self, i, data):
        """ Parse dipole moment components in [a.u.] 
        
        Returns
        -------
        numpy.ndarray
            Dipole moment in [a.u.]
        """
        match = re.search(self.hooks[self.func_name()], data[i])
        if match:
            mLogger.info("Dipole moment [a.u.]",
                         extra={"Parsed":V.dipole_moment})
            g = list(map(float,match.groups()))
            return np.asarray(g)

    @var_tag(V.diff_detach_mean)
    def diff_detach_mean(self, i, data):
        """ Parse mean position of detachment density [Ang] """
        mLogger.info("mean position of detachment density [Ang]",
                extra={"Parsed": V.diff_detach_mean})
        return parse_libwfa_vec("<r_h> [Ang]:", data, i)

    @var_tag(V.diff_attach_mean)
    def diff_attach_mean(self, i, data):
        """ Parse mean position of attachment density [Ang] """
        mLogger.info("mean position of attachment density [Ang]",
                extra={"Parsed": V.diff_attach_mean})
        return parse_libwfa_vec("<r_e> [Ang]:", data, i)

    @var_tag(V.diff_da_dist)
    def diff_da_dist(self, i, data):
        """ Parse linear D/A distance [Ang] """
        mLogger.info("linear D/A distance [Ang]",
                extra={"Parsed": V.diff_da_dist})
        return parse_libwfa_float("|<r_e - r_h>| [Ang]:", data, i)

    @var_tag(V.diff_detach_size)
    def diff_detach_size(self, i, data):
        """ Parse detachment size [Ang] """
        mLogger.info("RMS size of detachment density [Ang]",
                extra={"Parsed": V.diff_detach_size})
        return parse_libwfa_float("Hole size [Ang]:", data, i)

    @var_tag(V.diff_attach_size)
    def diff_attach_size(self, i, data):
        """ Parse attachment size [Ang] """
        mLogger.info("RMS size of attachment density [Ang]",
                extra={"Parsed": V.diff_attach_size})
        return parse_libwfa_float("Electron size [Ang]:", data, i)

    @var_tag(V.diff_detach_mom)
    def diff_detach_mom(self, i, data):
        """ Parse cartesian components of detachment size [Ang] """
        mLogger.info("cartesian components of detachment size [Ang]",
                extra={"Parsed": V.diff_detach_mom})
        j = 1
        vec = []
        while True:
            ls = data[i+j].split()
            if len(ls) == 0:
                break
            elif "Hole size [Ang]:" in data[i+j] and \
            "Cartesian components [Ang]:" in data[i+j+1]:
                vec = parse_inline_vec(data[i+j+1])
                break
            j += 1
        return vec

    @var_tag(V.diff_attach_mom)
    def diff_attach_mom(self, i, data):
        """ Parse cartesian components of attachment size [Ang] """
        mLogger.info("cartesian components of attachment size [Ang]",
                extra={"Parsed": V.diff_attach_mom})
        j = 1
        vec = []
        while True:
            ls = data[i+j].split()
            if len(ls) == 0:
                break
            elif "Electron size [Ang]:" in data[i+j] and \
            "Cartesian components [Ang]:" in data[i+j+1]:
                vec = parse_inline_vec(data[i+j+1])
                break
            j += 1
        return vec

    @var_tag(V.mulliken)
    def mulliken_adc(self, i, data):
        """ Parse MP(x) and ADC(x) mulliken charges """
        chg, n = [], 0
        while True:
            if "-------" in data[i+3+n]:
                break
            else:
                chg.append(data[i+3+n].split()[1:])
                n += 1
        chg = [[x[0]]+[float(x[1])] for x in chg]
        mLogger.info("MP(x)/ADC(x) Mulliken charges",
                     extra={"Parsed":V.mulliken})
        return chg

    @var_tag(V.trans_hole_mean)
    def trans_hole_mean(self, i, data):
        """ Parse mean position of hole [Ang] """
        mLogger.info("mean position of hole [Ang]",
                extra={"Parsed": V.trans_hole_mean})
        return parse_libwfa_vec("<r_h> [Ang]:", data, i)

    @var_tag(V.trans_elec_mean)
    def trans_elec_mean(self, i, data):
        """ Parse mean position of electron [Ang] """
        mLogger.info("mean position of electron [Ang]",
                extra={"Parsed": V.trans_elec_mean})
        return parse_libwfa_vec("<r_e> [Ang]:", data, i)

    @var_tag(V.trans_eh_dist)
    def trans_eh_dist(self, i, data):
        """ Parse linear e/h distance [Ang] """
        mLogger.info("linear e/h distance [Ang]",
                extra={"Parsed": V.trans_eh_dist})
        return parse_libwfa_float("|<r_e - r_h>| [Ang]:", data, i)

    @var_tag(V.trans_hole_size)
    def trans_hole_size(self, i, data):
        """ Parse RMS hole size [Ang] """
        mLogger.info("RMS hole size [Ang]",
                extra={"Parsed": V.trans_hole_size})
        return parse_libwfa_float("Hole size [Ang]:", data, i)

    @var_tag(V.trans_elec_size)
    def trans_elec_size(self, i, data):
        """ Parse RMS electron size [Ang] """
        mLogger.info("RMS electron size [Ang]",
                extra={"Parsed": V.trans_elec_size})
        return parse_libwfa_float("Electron size [Ang]:", data, i)

    @var_tag(V.trans_eh_sep)
    def trans_eh_sep(self, i, data):
        """ Parse RMS electron-hole separation [Ang] """
        mLogger.info("RMS electron size [Ang]",
                extra={"Parsed": V.trans_eh_sep})
        return parse_libwfa_float("RMS electron-hole separation", data, i)

    @var_tag(V.trans_hole_mom)
    def trans_hole_mom(self, i, data):
        """ Parse cartesian components of hole size [Ang] """
        mLogger.info("cartesian components of hole size [Ang]",
                extra={"Parsed": V.trans_hole_mom})
        j = 1
        vec = []
        while True:
            ls = data[i+j].split()
            if len(ls) == 0:
                break
            elif "Hole size [Ang]:" in data[i+j] and \
            "Cartesian components [Ang]:" in data[i+j+1]:
                vec = parse_inline_vec(data[i+j+1])
                break
            j += 1
        return vec

    @var_tag(V.trans_elec_mom)
    def trans_elec_mom(self, i, data):
        """ Parse cartesian components of electron size [Ang] """
        mLogger.info("cartesian components of electron size [Ang]",
                extra={"Parsed": V.trans_elec_mom})
        j = 1
        vec = []
        while True:
            ls = data[i+j].split()
            if len(ls) == 0:
                break
            elif "Electron size [Ang]:" in data[i+j] and \
            "Cartesian components [Ang]:" in data[i+j+1]:
                vec = parse_inline_vec(data[i+j+1])
                break
            j += 1
        return vec

    @var_tag(V.trans_eh_sep_mom)
    def trans_eh_sep_mom(self, i, data):
        """ Parse cartesian components of RMS
        electron-hole separation [Ang] """
        mLogger.info("RMS electron-hole separation [Ang]",
                extra={"Parsed": V.trans_eh_sep_mom})
        j = 1
        vec = []
        while True:
            ls = data[i+j].split()
            if len(ls) == 0:
                break
            elif "RMS electron-hole separation" in data[i+j] and \
            "Cartesian components [Ang]:" in data[i+j+1]:
                vec = parse_inline_vec(data[i+j+1])
                break
            j += 1
        return vec

    @var_tag(V.trans_eh_cov)
    def trans_eh_cov(self, i, data):
        """ Parse electron-hole covariance [Ang^2] """
        mLogger.info("electron-hole covariance [Ang^2]",
                extra={"Parsed": V.trans_eh_cov})
        return parse_libwfa_float("Covariance(r_h, r_e)", data, i)

    @var_tag(V.trans_eh_corr)
    def trans_eh_corr(self, i, data):
        """ Parse electron-hole correlation coefficient """
        mLogger.info("electron-hole correlation coefficient",
                extra={"Parsed": V.trans_eh_corr})
        return parse_libwfa_float("Correlation coefficient:", data, i)

class FDE_ADC(QCMethod):
    """ Parsing related to FDE-ADC implementation in Q-Chem """
    def __init__(self):
        super().__init__()# necessary for every derived class of QCMethod
        self.hooks = {"prepol_charges": "charges (prepol):",
                      "omega_ref": "FDE control parameter",
                      "omega_I": "Omega(FDE)",
                      "trust": "lambda(FDE)",
                      "delta_lin": "Delta_Lin:",
                      "timing": "FDE timings",
                      "expansion": "FDE-Expansion",
                      "method_Aref": "Method to generate rhoA_ref:",
                      "method_B": "Method to generate rhoB:",
                      "method_B_legacy": "Environment method",
                      "tfunc": "Embedding Pot. T functional",
                      "xcfunc": "Embedding Pot. XC functional",
                      "xfunc" : "Embedding Pot. X functional",
                      "cfunc" : "Embedding Pot. C functional",
                      "nonel_ref" : "LinFDET terms involving",
                      "subsysA"  : "Embedded System (A):",
                      "subsysB"  : "Environment (B):",
                      "E_xc_nad" : "non-additive E_xc:",
                      "Ts_nad"   : "non-additive T_s:",
                      "V_xc_nad_ref" : "# using -REFERENCE- potential",
                      "V_xc_nad"     : "# using -UPDATED- potential",
                      "V_t_nad_ref"  : "# using -REFERENCE- potential",
                      "V_t_nad"      : "# using -UPDATED- potential",
                      "J_int"        : "rho_A <-> rho_B:",
                      "AnucB"        : "rho_A <-> Nuc_B:",
                      "BnucA"        : "rho_B <-> Nuc_A:",
                      "V_NA_NB"      : "Nuc_A <-> Nuc_B:"}

    #TODO: implement new parsing for omega_ref, keep old one for legacy purpose
    def prepol_charges(self, i, data):
        """ Parse charges used for prepolarisation """
        self.add_variable(self.func_name(), V.prepol_charges)
        mLogger.info("charges used for prepolarisation",
             extra={"Parsed":V.prepol_charges})
        charges=[]
        i+=1
        while len(data[i].split())==1:
            charges.append(float(data[i]))
            i+=1
        return charges
    
    @var_tag(V.fde_omega_ref)
    def omega_ref(self, i, data):
        """ Parse FDE trust parameter (before construction of embedding potential) [ppm] """
        mLogger.info("a priori FDE overlap parameter Omega_ref",
                     extra={"Parsed":V.fde_omega_ref})
        return float(data[i].split()[5])

    @var_tag(V.fde_omega_I)
    def omega_I(self, i, data):
        """ Parse state-specific trust parameter Omega_I [ppm]
            from final FDE output """
        mLogger.info("state specific FDE overlap parameter Omega",
                     extra={"Parsed":V.fde_omega_I})
        return float(data[i].split()[1])

    @var_tag(V.fde_trust)
    def trust(self, i, data):
        """ Parse FDE trust parameter [ppm] from final FDE output """
        mLogger.info("state specific FDE overlap parameter Lambda",
                     extra={"Parsed":V.fde_trust})
        return float(data[i].split()[1])

    @var_tag(V.fde_delta_lin)
    def delta_lin(self, i, data):
        """ Parse 1st order term [eV] of LinFDET approximation """
        mLogger.info("1st order term of linearized FDET",
                     extra={"Parsed":V.fde_delta_lin})
        return float(data[i].split()[1])

    @var_tag(V.fde_timing)
    def timing(self, i, data):
        """ Parses FDE timings from the FDE summary.
        The general order of appearance is FDE-method, RhoA_ref, RhoB, v_emb,
        although RhoA_ref or RhoB might not be present due to import.

        Returns
        -------
        timings : dict
            A dictionary mapping tuples (CPU, wall) containing the times in seconds
            according to their label.
        """
        timings = {}
        pattern = r"(?P<label>\b.+\S+)\s+(?P<cpu>\d+\.\d+)\s+\(.+\)\s+(?P<wall>\d+\.\d+)"
        n = 0
        while True:
            if "------" in data[i+4+n]:
                break
            match = re.search(pattern, data[i+4+n])
            if match:
                lbl = match.group("label")
                timings[lbl] = (float(match.group("cpu")),
                                float(match.group("wall")))
            elif match == None:
                break
            n += 1

        mLogger.info("final FDE timings", extra={"Parsed":V.fde_timing})
        return timings

    @var_tag(V.fde_expansion)
    def expansion(self, i, data):
        """ Parses FDE-Expansion type [ME, SE, RADM]
        
        Returns
        -------
        string
            Expansion type
        """
        mLogger.info("FDET Basis set expansion",
                     extra={"Parsed":V.fde_expansion})
        return data[i].split()[1]

    @var_tag(V.fde_method_rhoA)
    def method_Aref(self, i, data):
        """ Parses which method is used to generate rhoAref (the density used
        to construct the initial embedding potential)
        
        So far the choices are "HF", "DFT / Func_Name" or "imported"
        
        Returns
        -------
        string
            RhoAref method
        """
        mLogger.info("FDET Method for rhoAref",
                     extra={"Parsed":V.fde_method_rhoA})
        s = data[i].split()
        if s[-1] == "'Densmat_A.txt')":
            return "imported"
        else:
            return s[-1]

    @var_tag(V.fde_method_rhoB)
    def method_B(self, i, data):
        """ Parses method type for rhoB. 
        
        So far the choices are "HF", "DFT / Func_Name" or "imported"
        
        Returns
        -------
        string
            RhoB method
        """
        mLogger.info("FDET Method for rhoB",
                     extra={"Parsed":V.fde_method_rhoB})
        s = data[i].split()
        if s[-1] == "'Densmat_B.txt')":
            return "imported"
        else:
            return s[-1]

    @var_tag(V.fde_method_rhoB)
    def method_B_legacy(self, i, data):
        """ Parses method type for rhoB. 
        
        So far the choices are "HF", "DFT / Func_Name" or "imported"
        
        Returns
        -------
        string
            RhoB method
        """
        mLogger.info("FDET Method for rhoB", extra={"Parsed":V.fde_method_rhoB})
        return data[i].split()[-1]

    @var_tag(V.fde_Tfunc)
    def tfunc(self, i, data):
        """ Determine what kinetic energy functional is used to construct
        the embedding potential.
        
        Returns
        -------
        string
            Abbreviation of kinetic energy functional
        """
        mLogger.info("kinetic energy functional for v_emb",
                     extra={"Parsed":V.fde_Tfunc})
        return data[i].split()[-1]

    @var_tag(V.fde_XCfunc)
    def xcfunc(self, i, data):
        """ Determine what exchange-correlation functional is used to construct
        the embedding potential.
        
        Returns
        -------
        string
            Abbreviation of exchange-correlation energy functional
        """
        mLogger.info("exchange-correlation functional for v_emb",
                     extra={"Parsed":V.fde_XCfunc})
        return data[i].split()[-1]

    @var_tag(V.fde_Xfunc)
    def xfunc(self, i, data):
        """ Determine what exchange functional is used to construct
        the embedding potential.
        
        Returns
        -------
        string
            Abbreviation of exchange energy functional
        """
        mLogger.info("exchange functional for v_emb",
                     extra={"Parsed":V.fde_XCfunc})
        return data[i].split()[-1]

    @var_tag(V.fde_Cfunc)
    def cfunc(self, i, data):
        """ Determine what correlation functional is used to construct
        the embedding potential.
        
        Returns
        -------
        string
            Abbreviation of correlation energy functional
        """
        mLogger.info("correlation functional for v_emb",
                     extra={"Parsed":V.fde_XCfunc})
        return data[i].split()[-1]

    @var_tag(V.fde_non_elstat_ref)
    def nonel_ref(self, i, data):
        """Parses non additive properties of the reference density
        
        Returns
        -------
        list
            List of [non-additive E_xc, non-additive T_s, integrated v_xc nad, integrated v_T nad]
        """
        l=[]
        mLogger.info("reference density non additive properties",
                     extra={"Parsed":V.fde_non_elstat_ref})
        l.append(float(data[i+2].split()[-2]))#non-additive E_xc
        l.append(float(data[i+3].split()[-2]))#non-additive T_s
        l.append(float(data[i+5].split()[-2]))#integrated v_xc nad
        l.append(float(data[i+6].split()[-2]))#integrated v_T nad
        return l

    @var_tag(V.fde_sysA)
    def subsysA(self, i, data):
        """Parses total energy of subsystem A WITHOUT embedding potential.
        
        Returns
        -------
        float
            Energy of subsystem A in [a.u.]
        """
        mLogger.info("non-additive exchange-correlation bi-functional",
                     extra={"Parsed":V.fde_sysA})
        return float(data[i].split()[-2])

    @var_tag(V.fde_sysB)
    def subsysB(self, i, data):
        """Parses total energy of subsystem B if calculated within fdeman.
        
        Returns
        -------
        float
            Energy of subsystem B in [a.u.]
        """
        mLogger.info("non-additive exchange-correlation bi-functional",
                     extra={"Parsed":V.fde_sysB})
        return float(data[i].split()[-2])

    @var_tag(V.fde_Exc_nad)
    def E_xc_nad(self, i, data):
        """Parses values of the non-additive exchange-correlation bi-functional
        
        Returns
        -------
        float
        """
        mLogger.info("non-additive exchange-correlation bi-functional",
                     extra={"Parsed":V.fde_Exc_nad})
        return float(data[i].split()[-2])

    @var_tag(V.fde_Ts_nad)
    def Ts_nad(self, i, data):
        """Parses values of the non-additive kinetic bi-functional
        
        Returns
        -------
        float
        """
        mLogger.info("non-additive kinetic bi-functional",
                     extra={"Parsed":V.fde_Ts_nad})
        return float(data[i].split()[-2])

    @var_tag(V.fde_int_xc_nad_ref)
    def V_xc_nad_ref(self, i, data):
        """Parses integral of some rhoA with the non-additive 
        exchange-correlation potential obtained with rhoA_ref:
        
        :math:`\\int \\rho_{A}\\cdot v_{xc}^{nad}[\\rho_{A}^{ref},
        \\rho_{B}](\\mathbf{r})`
        
        
        Returns
        -------
        float
        """
        mLogger.info(("expectation value of non-additive XC REFERENCE "
                      "potential"),
                     extra={"Parsed":V.fde_int_xc_nad_ref})
        return float(data[i+1].split()[-2])

    @var_tag(V.fde_int_xc_nad)
    def V_xc_nad(self, i, data):
        """Parses integral of some rhoA with the non-additive 
        exchange-correlation potential obtained with rhoA_emb (HF):
        
        :math:`\\int \\rho_{A}\\cdot v_{xc}^{nad}[\\rho_{A}^{emb,HF},
        \\rho_{B}](\\mathbf{r})`
        
        
        Returns
        -------
        float
        """
        mLogger.info(("expectation value of non-additive XC UPDATED "
                      "potential"),
                     extra={"Parsed":V.fde_int_xc_nad})
        return float(data[i+1].split()[-2])

    @var_tag(V.fde_int_Ts_nad_ref)
    def V_t_nad_ref(self, i, data):
        """Parses integral of some rhoA with the non-additive kinetic
        potential obtained with rhoA_ref:
        
        :math:`\\int \\rho_{A}\\cdot v_{xc}^{nad}[\\rho_{A}^{ref},
        \\rho_{B}](\\mathbf{r})`
        
        
        Returns
        -------
        float
        """
        mLogger.info(("expectation value of non-additive kinetic REFERENCE "
                      "potential"),
                     extra={"Parsed":V.fde_int_Ts_nad_ref})
        return float(data[i+2].split()[-2])

    @var_tag(V.fde_int_Ts_nad)
    def V_t_nad(self, i, data):
        """Parses integral of some rhoA with the non-additive kinetic
        potential obtained with rhoA_emb (HF):
        
        :math:`\\int \\rho_{A}\\cdot v_{xc}^{nad}[\\rho_{A}^{emb,HF},
        \\rho_{B}](\\mathbf{r})`
        
        
        Returns
        -------
        float
        """
        mLogger.info(("expectation value of non-additive kinetic UPDATED "
                      "potential"),
                     extra={"Parsed":V.fde_int_Ts_nad})
        return float(data[i+2].split()[-2])

    @var_tag(V.fde_J)
    def J_int(self, i, data):
        """Parses Coulomb repulsion of rhoA and rhoB.
        
         :math:`\\int\\int\\frac{\\rho_{A}(\\mathbf{r})\\rho_{B}(\\mathbf{r}')}
         {|\\mathbf{r}-\\mathbf{r}'|}\\mathrm{d}\\mathbf{r}'\\mathrm{d}
         \\mathbf{r}`
        
        Returns
        -------
        float
        """
        mLogger.info("Coulomb repulsion of rhoA-rhoB",
                     extra={"Parsed":V.fde_J})
        return float(data[i].split()[-2])

    @var_tag(V.fde_AnucB)
    def AnucB(self, i, data):
        """Parses Coulomb attraction of rhoA and NucB.
        
        :math:`\\int \\rho_{A}\\cdot v_B\\mathrm{d}
        \\mathbf{r}`
        
        """
        mLogger.info("Coulomb attraction of rhoA-nucB",
                     extra={"Parsed":V.fde_AnucB})
        return float(data[i].split()[-2])

    @var_tag(V.fde_BnucA)
    def BnucA(self, i, data):
        """Parses Coulomb attraction of rhoB and NucA.
        
        :math:`\\int \\rho_{B}\\cdot v_A\\mathrm{d}
        \\mathbf{r}`
        
        """
        mLogger.info("Coulomb attraction of rhoB-nucA",
                     extra={"Parsed":V.fde_BnucA})
        return float(data[i].split()[-2])

    @var_tag(V.fde_VNN)
    def V_NA_NB(self, i, data):
        """Parses Coulomb repulsion of nuclei NucA and NucB.
        
        :math:`\\sum_{A}\\sum_{B} \\frac{Z_{A}Z_{B}}{|\\mathbf{R}_{A} -
        \\mathbf{R}_{B}|}`
    
        """
        mLogger.info("Coulomb repulsion of nucA-nucB",
                     extra={"Parsed":V.fde_VNN})
        return float(data[i].split()[-2])

class CIS(QCMethod):
    """ Parsing related to FDE-ADC implementation in Q-Chem """
    def __init__(self):
        super().__init__()# necessary for every derived class of QCMethod
        self.hooks = {"exc_energies": (r"Excited state\s+(?P<state>\d+):\s*"
            r"excitation energy \(eV\)\s*=\s*(?P<energy>[-+]?\d+\.\d+)"),
            "adiabatic_center": (r"ADIABATIC CENTER\d+\s*LOCATIONS:.*"
            r"Occ\sx\s+([-]?\d+\.\d+)\sy\s+([-]?\d+\.\d+)\sz\s+([-]?\d+\.\d+) "
            r"Virt\sx\s+([-]?\d+\.\d+)\sy\s+([-]?\d+\.\d+)\sz\s+([-]?\d+\.\d+)"),
            "adiabatic_ss_dist": (r"ADIABATIC:.*sum of square of distances.*"
            r"Occ\s*(?P<occ>\d+\.\d+)\s*Virt\s*(?P<virt>\d+\.\d+)"),
            "adiabatic_hamilt": "Printing H in adiabatic representation",
            "diabatic_hamilt" : "Getting H in diabatic representation",
            "diabatic_center": (r"^DIABATIC CENTER\d+\s*LOCATIONS:.*"
            r"Occ\sx\s+([-]?\d+\.\d+)\sy\s+([-]?\d+\.\d+)\sz\s+([-]?\d+\.\d+) "
            r"Virt\sx\s+([-]?\d+\.\d+)\sy\s+([-]?\d+\.\d+)\sz\s+([-]?\d+\.\d+)"),
            "diabatic_ss_dist": (r"^DIABATIC:.*sum of square of distances.*"
            r"Occ\s*(?P<occ>\d+\.\d+)\s*Virt\s*(?P<virt>\d+\.\d+)")}

    @var_tag(V.exc_energy_rel)
    def exc_energies(self, i, data):
        """ Parse excitation energies [eV] """
        mLogger.info("relative CIS excitation energy/-ies [eV]",
                     extra={"Parsed": V.exc_energy_rel})
        match = re.search(self.hooks["exc_energies"], data[i])
        if match:
            return float(match.group("energy"))
    @var_tag(V.adia_center)
    def adiabatic_center(self, i, data):
        """ Parse adiabatic center """
        mLogger.info("adiabatic center locations",
                     extra={"Parsed": V.adia_center})
        match = re.search(self.hooks["adiabatic_center"], data[i])
        if match:
            all_coord = [float(c) for c in match.groups()]
            return [all_coord[:3], all_coord[3:]]

    @var_tag(V.adia_ss_dist)
    def adiabatic_ss_dist(self, i, data):
        """ Parse sum of square distances to other states """
        mLogger.info("adiabatic sum of squares of distance",
                     extra={"Parsed": V.adia_ss_dist})
        match = re.search(self.hooks[self.func_name()], data[i])
        if match:
            return [float(match.group("occ")), float(match.group("virt"))]

    @var_tag(V.adia_hamilt)
    def adiabatic_hamilt(self, i, data):
        """ Parse the adiabatic representation of the Hamiltonian """
        mLogger.info("adiabatic representation of the Hamiltonian",
                extra={"Parsed": V.adia_hamilt})
        j = 1
        H = []
        while True:
            ls = data[i+j].split()
            if not "showmatrix" in data[i+j]:
                break
            elif not "=" in data[i+j]:
                break
            elif len(ls) == 0:
                break
            H.append(float(ls[-1]))
            j += 1
        H = np.asarray(H)
        if is_square(len(H)):
            n = int(np.sqrt(len(H)))
            H = H.reshape((n,n))
        return H

    @var_tag(V.dia_hamilt)
    def diabatic_hamilt(self, i, data):
        """ Parse the diabatic representation of the Hamiltonian """
        mLogger.info("diabatic representation of the Hamiltonian",
                extra={"Parsed": V.dia_hamilt})
        j = 1
        H = []
        while True:
            ls = data[i+j].split()
            if not "showmatrix" in data[i+j]:
                break
            elif not "=" in data[i+j]:
                break
            elif len(ls) == 0:
                break
            H.append(float(ls[-1]))
            j += 1
        H = np.asarray(H)
        if is_square(len(H)):
            n = int(np.sqrt(len(H)))
            H = H.reshape((n,n))
        return H

    @var_tag(V.dia_center)
    def diabatic_center(self, i, data):
        """ Parse diabatic center """
        mLogger.info("adiabatic center locations",
                     extra={"Parsed": V.dia_center})
        match = re.search(self.hooks[self.func_name()], data[i])
        if match:
            all_coord = [float(c) for c in match.groups()]
            return [all_coord[:3], all_coord[3:]]

    @var_tag(V.dia_ss_dist)
    def diabatic_ss_dist(self, i, data):
        """ Parse sum of square distances to other states """
        mLogger.info("diabatic sum of squares of distance",
                     extra={"Parsed": V.dia_ss_dist})
        match = re.search(self.hooks[self.func_name()], data[i])
        if match:
            return [float(match.group("occ")), float(match.group("virt"))]
