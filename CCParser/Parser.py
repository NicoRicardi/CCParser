import importlib as il
import os.path
import inspect
import re
import logging
import json
from .ParserData import Struct
from .ParserData import ParseContainer
from .ParserData import StructEncoder
from .QCBase import GenFormatter, VarNames as V

class Parser(object):
    def __init__(self, output, *, software=None, to_console=True,
                 to_file=False, log_file="CCParser.log", to_json=False,
                 json_file="CCParser.json"):#cf. PEP-3102
        """ Parser constructor.

        Parameters
        ----------
        output : string
            Output filename.
        software : string
            Name of quantum chemistry software suite (default: None).
        to_console : bool
            Whether to print log output to screen (default: True).
        to_file : bool
            Whether to write log output to file (default: False).
        log_file : string
            Name of output log file (default: ``CCParser.log``).
        to_json : bool
            Whether to dump CCParser.results to JSON file.
        json_file : string
            Name of JSON output file.
        """
        self.f_output = output
        self.logger = logging.getLogger("CCParser")
        self.to_console = to_console
        self.to_file = to_file
        self.logname = log_file
        self.setupLogger()

        if software != None:
            self.software = software
        else:
            self.find_software()
            raise ValueError("No software specified!")

        self.output_basename = os.path.basename(output)
        self.read_output()# read output to memory
        self.load_methods()# software dependent import
        self.results = Struct()# Set up container
        self.logger.warning("CCParser starts...")
        for i, line in enumerate(self.rawData):
            for mthd in self.methods:
#                match, key = self.canParse(line, mthd)
                match, keys = self.canParse(line, mthd)
                if match:
                    for key in keys:# if not 1-to-1 mapping
                        q = self.get_quantity(i, key, mthd)
                        if hasattr(self.results, mthd.map[key]):
                            obj = getattr(self.results, mthd.map[key])
                            obj.add(i, q)
                        else:
                            obj = ParseContainer()
                            obj.add(i, q)
                            setattr(self.results, mthd.map[key], obj)
        if not hasattr(self.results, V.has_finished):
            container = ParseContainer()
            container.add(0, False)
            setattr(self.results, V.has_finished, container)
            self.logger.warning("Output indicates abnormal exit. Added "+
                                "[results.has_finished] = False")
        if to_json:
            self.dump_json(fname=json_file)
        self.logger.warning("CCParser has finished.")
        self.loggerCleanUp()

    def read_output(self):
        """ Read in output file """
        with open(self.f_output, "r") as f:
            self.rawData = f.readlines()

    def read_input(self, f_input):
        """ (Optional) Read input file """
        with open(f_input) as n:
            self.rawData.insert(0, n.readlines())

    def canParse(self, line, mthd):
        """ Check if line is parsable """
        found = False
        keys = []#for cases where there's no 1-to-1 mapping
        for key, value in mthd.hooks.items():
            if value in line:
                found = True
                keys.append(key)
#                return found, key
            else:
                match = re.search(value, line)
                if match:
                    found = True
                    keys.append(key)
#                    return found, key
        if not found:
            return found, None
        else:
            return found, keys

    def get_quantity(self, i, key, mthd):
        """ Call function of method class. This is the actual parsing. """
        method_func = getattr(mthd, key)# needs to be method not list of methods
        result = method_func(i, self.rawData)
        return result

    def load_methods(self):
        """ Load correct module which contains parsing information
        based on which software was specified. """
        tmp = re.sub('[^A-Za-z]+', '', self.software.lower())
        if tmp == "qchem":
            m_package = ".QChem"
        elif tmp == "gaussian":
            m_package = ".Gaussian"
        elif tmp == "molcas":
            raise NotImplementedError("Molcas parsing not implemented yet!")
            m_package = ".Molcas"
        elif tmp == "turbomole":
            raise NotImplementedError("Turbomole parsing not implemented yet!")
            m_package = ".Turbomole"
        elif tmp == "psi":
            m_package = ".Psi4"
        else:
            raise ValueError("The specified software is misspelled or not implemented yet!")
        #global m
        m = il.import_module(m_package, package="CCParser")
        self.method_names = [k[0] for k in inspect.getmembers(m,\
            inspect.isclass) if k[1].__module__ == "CCParser"+m_package]
        # this also instantiates!!
        self.methods = [getattr(m, mname)() for mname in self.method_names]

    def setupLogger(self):
        """Initiate logger for CCParser.Parser"""
        # Set main logger's minimum output level
        self.logger.setLevel(logging.INFO)
        # Set up Formatter
#        p_fmt = logging.Formatter("[results.%(Parsed)s] Parsed %(message)s")
#
        # This is abusing the Formatter class a bit, but I wanted to avoid
        # one Logger for every format, maybe I'll change this in the future.
        p_fmt = GenFormatter(
            {logging.INFO: "[results.%(Parsed)s] Parsed %(message)s",
             logging.WARNING: "==[%(asctime)s]== %(message)s",
             logging.ERROR: "%(message)s"})
        # Set up Handlers
        if self.to_file:
            fh = logging.FileHandler(self.logname)
            fh.setLevel(logging.INFO)
            fh.setFormatter(p_fmt)
            self.logger.addHandler(fh)
        if self.to_console:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(p_fmt)
            self.logger.addHandler(ch)
        # No output in case both booleans are False
        if not any([self.to_console, self.to_file]):
            self.logger.setLevel(logging.CRITICAL)

    def loggerCleanUp(self):
        """In order to avoid multiplying handlers. """
        for i in range(len(self.logger.handlers)):
            self.logger.handlers.pop()

    def set_missing_keys(self):
        """Set default values for keywords that have not been found."""
        # use V.fde_expansion as an indicaotr whether or not an FDE calculation
        # was requested
#        if hasattr(self.results, V.fde_expansion):
#            if not hasattr(self.results, V.fde_isA_imported):
#                container = ParseContainer(0, False)
#                setattr(self.results, V.fde_isA_imported, container)
#                self.logger.info("whether FDET program imports rhoA_ref",
#                     extra={"Parsed":V.fde_isA_imported})
#            if not hasattr(self.results, V.fde_isB_imported):
#                container = ParseContainer(0, False)
#                setattr(self.results, V.fde_isB_imported, container)
#                self.logger.info("whether FDET program imports rhoB",
#                     extra={"Parsed":V.fde_isB_imported})
        if not hasattr(self.results, V.has_finished):
            container = ParseContainer(0, False)
            setattr(self.results, V.has_finished, container)
            self.logger.warning("Output indicates abnormal exit.")

    def dump_json(self, fname="CCParser.json"):
        """Dumps contens of the CCParser.results container to a JSON file.

        Parameters
        ----------
        fname : str
            Filename to dump to.
        """
        with open(fname, "w") as pdump:
            json.dump(self.results, pdump, cls=StructEncoder)
        self.logger.warning("Dumped CCParser.results to JSON file.")

    def find_software(self):
        pass