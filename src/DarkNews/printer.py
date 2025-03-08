import os
import pandas as pd
import numpy as np
import dill

from pathlib import Path
from particle import literals as lp

import DarkNews as dn
from DarkNews import const
from DarkNews import pdg
from DarkNews import Cfourvec as Cfv

if dn.HAS_PYHEPMC3:
    import pyhepmc as hep
    from pyhepmc import io

import logging

logger = logging.getLogger("logger." + __name__)
prettyprinter = logging.getLogger("prettyprinter." + __name__)


def print_in_order(x):
    return " ".join(f"{t:.8E}" for t in list(x[1:]))


class Printer:
    def __init__(self, df_gen, data_path=None, print_to_float32=False, decay_product="e+e-", sparse=0):
        """
        Main printer of DarkNews. Can print events contained in the pandas dataframe to files of various types.

        Args:
            df_gen (pd.DataFrame): dataframe with all generated events.

            data_path (str, optional): path to be used to save the event files. Defaults to the "data_path" attribute of df_gen

            print_to_float32 (bool, optional):  If true downgrade floats to float32 to save storage space.
                                                Only relevant when sparse > 2.
                                                Defaults to False.

            decay_product (str, optional):  what decay products are being used. Choices = ["e+e-", "mu+mu-", "photon"]. Defaults to 'e+e-'.

            sparse (int, optional): level of sparseness of dataframe.

        """

        # main DataFrame
        self.df_gen = df_gen
        self.decay_product = decay_product
        self.sparse = int(sparse)  # backwards compatibility
        self.print_to_float32 = print_to_float32

        self.unweighted_entries = None
        # sample size (# of events)
        self.tot_events = len(self.df_gen.index)

        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self.df_gen.attrs["data_path"]

        # file name and path (without extension)
        self.out_file_name = self.create_dir()

        self._kinematics_in_np_arrays = False

        self.particles_per_event = 7
        if self.decay_product == "photon":
            self.particles_per_event -= 1  # ell ell --> photon
        if self.sparse >= 2:  # lose target, recoil, and daughter HNL/nu
            self.particles_per_event -= 3
        if self.sparse == 4:  # lose parent HNL
            self.particles_per_event -= 1

    # Create target directory if it doesn't exist
    def create_dir(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        return Path(self.data_path)

    def print_events_to_ndarray(self, **kwargs):
        """
        Print to numpy array file (.npy)

        """

        if self.sparse >= 1:
            if self.print_to_float32:
                self.array_gen = self.df_gen.to_numpy(dtype=np.float32)
            else:
                self.array_gen = self.df_gen.to_numpy(dtype=np.float64)
            # cols = [f"{v[0]}_{v[1]}" if v[1] else f"{v[0]}" for v in self.df_gen.columns.values]
        else:
            if self.print_to_float32:
                logger.warning("WARNING! Can only downgrade dataframe to float32 when sparse >= 1. Proceeding with float64 instead.")

            # convert to numeric values
            self.df_gen = self.df_gen.replace(to_replace="conserving", value="+1")
            self.df_gen = self.df_gen.replace(to_replace="flipping", value="-1")

            # remove non-numeric entries
            self.df_for_numpy = self.df_gen.drop(["underlying_process", "target", "scattering_regime"], axis=1, level=0)
            # cols = [f"{v[0]}_{v[1]}" if v[1] else f"{v[0]}" for v in self.df_for_numpy.columns.values]
            self.array_gen = self.df_for_numpy.to_numpy(dtype=np.float64)

        filename = Path(f"{self.out_file_name}/ndarray.npy").__str__()
        np.save(filename, self.array_gen, allow_pickle=False, **kwargs)
        prettyprinter.info(f"Events in numpy array saved to file successfully:\n{filename}")
        return self.array_gen

    def print_events_to_parquet(self, **kwargs):
        """
        Print to pandas DataFrame to parquet file using fastparquet (.parquet)

        This format cannot save df.attrs to file.

        """

        filename = Path(f"{self.out_file_name}/pandas_df.parquet").__str__()

        # import pyarrow.parquet as pq
        # import pyarrow as pa
        # kwargs['engine']=kwargs.get('engine','pyarrow')
        dn.pq.write_table(dn.pa.Table.from_pandas(self.df_gen), filename, **kwargs)
        prettyprinter.info(f"Events in pandas dataframe (sparse = {self.sparse}) saved to parquet file successfully:\n{filename}")
        return self.df_gen

    def print_events_to_pandas(self, **kwargs):
        """
        Print to pandas DataFrame pickle file (.pckl)

        This is the only format that allows to save df.attrs to file.
        Using Dill to serialize the Model, Detector, and NuclearTarget classes to file.

        """
        filename = Path(f"{self.out_file_name}/pandas_df.pckl").__str__()

        if self.print_to_float32:
            if self.sparse >= 1:
                df = self.df_gen.apply(pd.to_numeric, downcast="float")
            else:
                df = self.df_gen
                logger.warning("WARNING! Can only downgrade dataframe to float32 when sparse >= 1. Proceeding with float64 instead.")
        else:
            df = self.df_gen

        # pickles DarkNews classes with support for lambda functions
        dill.dump(df, open(filename, "wb"), **kwargs)
        prettyprinter.info(f"Events in pandas dataframe saved to file successfully:\n{filename}")
        return df

    def get_unweighted_events(self, nevents, prob_col="w_event_rate", **kwargs):
        """
        Unweigh events in dataframe down to "nevents" using accept-reject method with the weights in "prob_col" of dataframe.

        Do this only once, unless nevents changes.

        Fails if nevents is smaller than the total number of unweighted events.

        kwargs passed to numpy's random.choice.
        """
        if self.unweighted_entries is None or nevents != len(self.unweighted_entries):
            logger.info(f"Unweighing events down to {nevents} entries.")
            prob = self.df_gen[prob_col] / np.sum(self.df_gen[prob_col])
            if (prob < 0).any():
                logger.error(f"ERROR! Probabily for unweighting contains negative values! Bad weights? {prob_col} < 0.")
            if (prob == 0).any():
                logger.warning(
                    f"WARNING! Discarding zero-valued weights for unweighting. Total of {sum(prob == 0)} of {len(prob)} zero entries for {prob_col}."
                )
                self.unweighted_entries = np.random.choice(self.df_gen.index[prob > 0], size=nevents, p=prob[prob > 0], *kwargs)
            else:
                self.unweighted_entries = np.random.choice(self.df_gen.index, size=nevents, p=prob, *kwargs)

        return self.unweighted_entries

        # # Unweigh events down to unweighted_hep_events?
        # if hep_unweight:
        #     df_gen = self.get_unweighted_events(nevents=unweighted_hep_events)
        # else:
        #     df_gen = self.df_gen

        # # extract selected entries
        # return self.df_gen.filter(AccEntries, axis=0).reset_index()

    def _prepare_kinematics(self, hep_unweight=False, unweighted_hep_events=None):
        """pre compute the numpy arrays from dataframe to speed up hepmc priting routines.

        Args:
            hep_unweight (bool, optional): If True, unweight event generation. Defaults to False.
            unweighted_hep_events (int, optional):  How many unweighted events to keep.
                                                    If None, do not unweight.
                                                    Should be a positive integer and much smaller than the total number of events.
        """

        if not self._kinematics_in_np_arrays:

            if "pos_scatt" not in self.df_gen.columns:
                p = "pos_scatt"
                self.df_gen[[(p, "0"), (p, "1"), (p, "2"), (p, "3")]] = 0.0

            if "pos_decay" not in self.df_gen.columns:
                p = "pos_decay"
                self.df_gen[[(p, "0"), (p, "1"), (p, "2"), (p, "3")]] = 0.0

            self.mass_projectile = Cfv.mass(self.df_gen["P_projectile"].to_numpy())
            self.mass_projectile[np.isnan(self.mass_projectile)] = 0.0  # Bad sqrt of small number
            self.pvec_projectile = self.df_gen["P_projectile"].to_numpy()

            # pre-computing some variables
            # if self.sparse <= 3:
            self.mass_decay_N_parent = Cfv.mass(self.df_gen["P_decay_N_parent"].to_numpy())
            self.pvec_decay_N_parent = self.df_gen["P_decay_N_parent"].to_numpy()

            if self.sparse <= 2:
                self.mass_recoil = Cfv.mass(self.df_gen["P_recoil"].to_numpy())
                self.pvec_recoil = self.df_gen["P_recoil"].to_numpy()

                self.mass_target = Cfv.mass(self.df_gen["P_target"].to_numpy())
                self.pvec_target = self.df_gen["P_target"].to_numpy()

            if self.sparse <= 1:
                self.mass_decay_N_daughter = Cfv.mass(self.df_gen["P_decay_N_daughter"].to_numpy())
                self.mass_decay_N_daughter[np.isnan(self.mass_decay_N_daughter)] = 0.0  # Bad sqrt of small number
                self.pvec_decay_N_daughter = self.df_gen["P_decay_N_daughter"].to_numpy()

            if self.decay_product == "e+e-" or self.decay_product == "mu+mu-":
                self.pvec_decay_ell_minus = self.df_gen["P_decay_ell_minus"].to_numpy()
                self.pvec_decay_ell_plus = self.df_gen["P_decay_ell_plus"].to_numpy()
            if self.decay_product == "photon":
                self.pvec_decay_photon = self.df_gen["P_decay_photon"].to_numpy()

            self.pvec_pos_decay = self.df_gen["pos_decay"].to_numpy()
            self.pvec_pos_scatt = self.df_gen["pos_scatt"].to_numpy()

            # string to be saved to file
            self.projectile_flavor = self.df_gen["projectile_pdgid"]

            if self.decay_product == "e+e-":
                self.id_lepton_minus = int(lp.e_minus.pdgid)
                self.id_lepton_plus = int(lp.e_plus.pdgid)
                self.lepton_mass = const.m_e
            elif self.decay_product == "mu+mu-":
                self.id_lepton_minus = int(lp.mu_minus.pdgid)
                self.id_lepton_plus = int(lp.mu_plus.pdgid)
                self.lepton_mass = const.m_mu
            elif self.decay_product == "photon":
                pass
            else:
                logger.warning(f"Decay product {self.decay_product} not recognized, assuming it to be e+e-.")
                self.id_lepton_minus = int(lp.e_minus.pdgid)
                self.id_lepton_plus = int(lp.e_plus.pdgid)
                self.lepton_mass = const.m_e

            # kinematics converted already
            self._kinematics_in_np_arrays = True

    def print_events_to_hepevt(self, filename=None, hep_unweight=False, unweighted_hep_events=100):
        if not hep_unweight:
            logger.warning(
                "WARNING: HEPevt is not a lossless format -- you will lose the event weights. \
If you want to force-print weights, use the hepevt_legacy format instead. \
Otherwise, please set hep_unweight=True and set the desired number of unweighted events."
            )
        # HEPevt file name
        if filename:
            hep_path = filename
        else:
            hep_path = Path(f"{self.out_file_name}/HEPevt.dat").__str__()
        if dn.HAS_PYHEPMC3:
            self._pyhepmc_printer(io.WriterHEPEVT(hep_path), hep_unweight=hep_unweight, unweighted_hep_events=unweighted_hep_events)
        else:
            logger.error("ERROR! Pyhepmc is not available. Please install it to print to HEPevt format")

    def print_events_to_hepmc2(self, filename=None, hep_unweight=False, unweighted_hep_events=100):
        # HEPevt file name
        if filename:
            hep_path = filename
        else:
            hep_path = Path(f"{self.out_file_name}/hep_ascii.hepmc2").__str__()
        if dn.HAS_PYHEPMC3:
            self._pyhepmc_printer(io.WriterAsciiHepMC2(hep_path), hep_unweight=hep_unweight, unweighted_hep_events=unweighted_hep_events)
        else:
            logger.error("ERROR! Pyhepmc is not available. Please install it to print to HEPmc2 format")

    def print_events_to_hepmc3(self, filename=None, hep_unweight=False, unweighted_hep_events=100):
        # HEPevt file name
        if filename:
            hep_path = filename
        else:
            hep_path = Path(f"{self.out_file_name}/hep_ascii.hepmc3").__str__()

        if dn.HAS_PYHEPMC3:
            self._pyhepmc_printer(io.WriterAscii(hep_path), hep_unweight=hep_unweight, unweighted_hep_events=unweighted_hep_events)
        else:
            logger.error("ERROR! Pyhepmc is not available. Please install it to print to HEPevt format")

    def _pyhepmc_printer(self, hep_writer, hep_unweight=False, unweighted_hep_events=100):
        """Use pyhepmc to print events to standard HEP formats.

        Args:
            hep_writer (an instance of a hep writer bindings): of one of the following types: WriterHEPEVT, WriterAsciiHepMC2, WriterAscii
            hep_unweight (bool, optional): if true, unweight events. Defaults to False.
            unweighted_hep_events (int, optional): if hep_unweight is true, use this value to determine how many unweighted events to print. Defaults to 100.

        Events are printed using the numpy arrays that have obtained from the pandas dataframe. This speeds up the printing process
        Four momenta are printed following the convention:

            px py pz e pdgid status_code

            status_code is defined as:
                0 Not defined (null entry) Not a meaningful status
                1 Undecayed physical particle Recommended for all cases
                2 Decayed physical particle Recommended for all cases
                3 Documentation line Often used to indicate
                in/out particles in hard process
                4 Incoming beam particle Recommended for all cases
                5-10 Reserved for future standards Should not be used
                11-200 Generator-dependent For generator usage
                201- Simulation-dependent For simulation software usage


        """

        # pre-compute kinematics with numpy for faster index access
        self._prepare_kinematics()

        # converting Lorentz order -- px, py, pz, E
        hep_order = [1, 2, 3, 0]

        ri = hep.GenRunInfo()
        from . import __version__

        ri.tools = [("DarkNews", __version__, "DarkNews upscattering engine")]

        # name of the weights -- w_event_rate, w_flux_avg_xsec, w_decay_rate_0 (, w_decay_rate_1)
        df_weight_names = [name for name in list(self.df_gen.columns.levels[0]) if "w_" in name]
        ri.weight_names = df_weight_names

        if hep_unweight:
            events = self.get_unweighted_events(unweighted_hep_events)
        else:
            events = self.df_gen.index

        # loop over events
        for counter, i in enumerate(events):
            evt = hep.GenEvent(hep.Units.GEV, hep.Units.CM)
            evt.event_number = counter

            v1 = hep.GenVertex((self.pvec_pos_scatt[i, hep_order]))
            v2 = hep.GenVertex((self.pvec_pos_decay[i, hep_order]))

            # SCATTER
            # Upscattering process
            p1 = hep.GenParticle(self.pvec_projectile[i, hep_order], self.projectile_flavor[i], 4)
            p1.generated_mass = self.mass_projectile[i]
            evt.add_particle(p1)
            v1.add_particle_in(p1)

            if self.sparse <= 1:
                p2 = hep.GenParticle(self.pvec_target[i, hep_order], int(self.df_gen["target_pdgid", ""].to_numpy()[i]), 4)
                p2.generated_mass = self.mass_target[i]
                evt.add_particle(p2)
                v1.add_particle_in(p2)

            p3 = hep.GenParticle(self.pvec_decay_N_parent[i, hep_order], int(pdg.neutrino5.pdgid), 2)
            p3.generated_mass = self.mass_decay_N_parent[i]
            evt.add_particle(p3)
            v1.add_particle_out(p3)
            v2.add_particle_in(p3)

            if self.sparse <= 1:
                p4 = hep.GenParticle(self.pvec_recoil[i, hep_order], int(self.df_gen["target_pdgid", ""].to_numpy()[i]), 2)
                p4.generated_mass = self.mass_recoil[i]
                evt.add_particle(p4)
                v1.add_particle_out(p4)

            # DECAY
            if self.sparse <= 1:
                pnu = hep.GenParticle(self.pvec_decay_N_daughter[i, hep_order], int(pdg.neutrino4.pdgid), 1)
                pnu.generated_mass = self.mass_decay_N_daughter[i]
                evt.add_particle(pnu)
                v2.add_particle_out(pnu)

            if self.decay_product == "e+e-" or self.decay_product == "mu+mu-":
                pep = hep.GenParticle(self.pvec_decay_ell_plus[i, hep_order], int(self.id_lepton_plus), 1)
                pep.generated_mass = self.lepton_mass
                evt.add_particle(pep)

                pem = hep.GenParticle(self.pvec_decay_ell_minus[i, hep_order], int(self.id_lepton_minus), 1)
                pem.generated_mass = self.lepton_mass
                evt.add_particle(pem)

                v2.add_particle_out(pep)
                v2.add_particle_out(pem)

            elif self.decay_product == "photon":
                pphoton = hep.GenParticle(self.pvec_decay_photon[i, hep_order], int(lp.photon.pdgid), 1)
                pphoton.generated_mass = 0.0
                v2.add_particle_out(pphoton)

            # now add the scattering and decay vertices
            evt.add_vertex(v1)
            evt.add_vertex(v2)

            evt.run_info = ri
            if not hep_unweight:
                evt.weights = [self.df_gen[name, ""].to_numpy()[i] for name in df_weight_names]

            # write event to file using the chosen hep writer (HEPevt, hepmc2 or 3)
            hep_writer.write(evt)

    def print_events_to_hepevt_legacy(self, filename=None, hep_unweight=False, unweighted_hep_events=100):
        """
        Print events to HEPevt format.

            The file start with the total number of events:

                'tot_events_to_print'

            On a new line, each event starts with a brief description of the event:

                'event_number number_of_particles (event_weight if it exists)'

            On a new line, a new particle and its properties are added. Using string concatenation,
            we print each particle as follows:

            (
                f'0 '				  	# ignored = 0 or tracked = 1
                f' {i}'				  	# particle PDG number
                f' 0 0 0 0' 		  	# ????? (parentage)
                f' {*list([px,py,pz])}'	# particle px py pz momenta
                f' {E}' 				# particle energy
                f' {m}'				  	# particle mass
                f' {*list([x,y,z])}' 	# spatial x y z coords
                f' {t}' 				# time coord
                '\n'
            )

            The last two steps repeat until the EOF.

        """

        # pre-compute kinematics with numpy for faster index access
        self._prepare_kinematics(hep_unweight=hep_unweight, unweighted_hep_events=unweighted_hep_events)

        logger.info("Printing events to HEPevt using legacy format...")
        lines = []

        if hep_unweight:
            events = self.get_unweighted_events(unweighted_hep_events)
        else:
            events = self.df_gen.index

        # loop over events
        for counter, i in enumerate(events):
            # no particles & event id
            if hep_unweight:
                lines.append(f"{counter} {self.particles_per_event}\n")
            else:
                lines.append(f"{counter} {self.particles_per_event} {self.df_gen['w_event_rate',''].to_numpy()[i]:.8E}\n")

            if self.sparse < 4:
                # scattering inital states
                lines.append(
                    (  # Projectile
                        f"0 "
                        f" {self.projectile_flavor[i]}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_projectile[i])}"
                        f" {self.df_gen['P_projectile','0'].to_numpy()[i]:.8E}"
                        f" {self.mass_projectile[i]:.8E}"
                        f" {print_in_order(self.pvec_pos_scatt[i])}"
                        f" {self.df_gen['pos_scatt','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

            if self.sparse <= 1:
                lines.append(
                    (  # Target
                        f"0 "
                        f" {int(self.df_gen['target_pdgid',''].to_numpy()[i])}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_target[i])}"
                        f" {self.df_gen['P_target','0'].to_numpy()[i]:.8E}"
                        f" {self.mass_target[i]:.8E}"
                        f" {print_in_order(self.pvec_pos_scatt[i])}"
                        f" {self.df_gen['pos_scatt','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

            # scatter final products
            if self.sparse <= 3:
                lines.append(
                    (  # HNL produced
                        f"0 "
                        f" {int(pdg.neutrino5.pdgid)}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_decay_N_parent[i])}"
                        f" {self.df_gen['P_decay_N_parent','0'].to_numpy()[i]:.8E}"
                        f" {self.mass_decay_N_parent[i]:.8E}"
                        f" {print_in_order(self.pvec_pos_scatt[i])}"
                        f" {self.df_gen['pos_scatt','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

            if self.sparse <= 1:
                lines.append(
                    (  # recoiled target
                        f"0 "
                        f" {int(self.df_gen['target_pdgid',''].to_numpy()[i])}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_recoil[i])}"
                        f" {self.df_gen['P_recoil','0'].to_numpy()[i]:.8E}"
                        f" {self.mass_recoil[i]:.8E}"
                        f" {print_in_order(self.pvec_pos_scatt[i])}"
                        f" {self.df_gen['pos_scatt','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

                # decay final products
                lines.append(
                    (  # daughter neutrino/HNL
                        f"0 "
                        f" {int(pdg.nulight.pdgid)}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_decay_N_daughter[i])}"
                        f" {self.df_gen['P_decay_N_daughter','0'].to_numpy()[i]:.8E}"
                        f" {self.mass_decay_N_daughter[i]:.8E}"
                        f" {print_in_order(self.pvec_pos_decay[i])}"
                        f" {self.df_gen['pos_decay','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

            # Always print visible decay products
            if self.decay_product == "e+e-" or self.decay_product == "mu+mu-":
                lines.append(
                    (  # ell-
                        f"1 "
                        f" {self.id_lepton_minus}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_decay_ell_minus[i])}"
                        f" {self.df_gen['P_decay_ell_minus','0'].to_numpy()[i]:.8E}"
                        f" {self.lepton_mass:.8E}"
                        f" {print_in_order(self.pvec_pos_decay[i])}"
                        f" {self.df_gen['pos_decay','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )
                lines.append(
                    (  # ell+
                        f"1 "
                        f" {self.id_lepton_plus}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_decay_ell_plus[i])}"
                        f" {self.df_gen['P_decay_ell_plus','0'].to_numpy()[i]:.8E}"
                        f" {self.lepton_mass:.8E}"
                        f" {print_in_order(self.pvec_pos_decay[i])}"
                        f" {self.df_gen['pos_decay','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )

            elif self.decay_product == "photon":
                lines.append(
                    (  # photon
                        f"1 "
                        f" {pdg.photon.pdgid}"
                        f" 0 0 0 0"
                        f" {print_in_order(self.pvec_decay_photon[i])}"
                        f" {self.df_gen['P_decay_photon','0'].to_numpy()[i]:.8E}"
                        f" 0"
                        f" {print_in_order(self.pvec_pos_decay[i])}"
                        f" {self.df_gen['pos_decay','0'].to_numpy()[i]:.8E}"
                        "\n"
                    )
                )
            else:
                raise RuntimeError(f"Could not print to HEPevt legacy format. Decay product: {self.decay_product} not recognized.")

        # string to be saved to file
        hepevt_string = ""
        hepevt_string += f"{len(events)}\n"

        # HEPevt file name
        if filename:
            hepevt_file_name = Path(filename).__str__()
        else:
            hepevt_file_name = Path(f"{self.out_file_name}/HEPevt_legacy.dat").__str__()

        f = open(hepevt_file_name, "w+")
        # print events
        f.write(hepevt_string + "".join(lines))
        f.close()

        prettyprinter.info(f"HEPevt events saved to file successfully:\n{hepevt_file_name}")
