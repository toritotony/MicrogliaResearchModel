from __future__ import annotations
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


class Neuron(Agent):
    """Neuron agent: healthy ↔ damaged → dead."""
    def __init__(
        self,
        unique_id: int,
        model,
        damaged: bool = False,
        dead: bool = False,
        healthy: bool = True,
        reach: int = 2,
        ticks_damaged: int = 0,
    ):
        """Initialize neuron with state and parameters.

        Args:
            unique_id (int): unique identifier of agent
            model (Object): reference to the simulation model
            damaged (bool, optional): is the neuron damaged. Defaults to False.
            dead (bool, optional): is the neuron dead. Defaults to False.
            healthy (bool, optional): is the neuron healthy. Defaults to True.
            reach (int, optional): how much reach does the neuron have to sense its surroundings. Defaults to 2.
            ticks_damaged (int, optional): how many ticks the neuron has been damaged. Defaults to 0.
            
        Internal state variables:
            lipid_droplets (float): represents the accumulation of lipid droplets in the neuron, which can influence its health and interactions with glial cells. Defaults to 0.0.
            toxic_lipids (float): represents the accumulation of toxic lipid species in the neuron, which can arise from mitochondrial dysfunction or uptake from LDAM microglia, and can drive damage and inflammation. Defaults to 0.0.
        """
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.damaged: bool = damaged
        self.dead: bool = dead
        self.healthy: bool = healthy and not (damaged or dead)
        self.reach: int = reach
        self.ticks_damaged: int = ticks_damaged
        self.lipid_droplets: float = 0.0
        self.toxic_lipids: float = 0.0 # From LDAM Microglia

        if self.damaged:
            self.healthy = False
        if self.dead:
            self.healthy = False
            self.damaged = False

    def become_damaged(self):
        """ Transition neuron to damaged state. Updates inflmamation value, damage value, reach, and lipid droplets."""
        if self.dead or self.damaged or self.pos is None:
            return
        self.healthy = False
        self.damaged = True
        self.ticks_damaged = 0
        x, y = self.pos
        self.model.residence[x, y] = True
        self.model.res_curr[x, y] = max(self.model.res_curr[x, y], 1)
        # neuron-sourced pro-inflam & damage (chemotaxis source), with boost from lipid droplets (e.g., from mitochondrial dysfunction), and less reach to sense anti-inflam signals and recover
        self.model.pro_inflam_val[x, y] = max(self.model.pro_inflam_val[x, y], 1.0)
        self.model.damage_val[x, y] = max(self.model.damage_val[x, y], 1.0)
        self.reach = 1
        self.lipid_droplets = min(
            self.lipid_droplets + self.model.neuron_ld_damage_boost,
            self.model.neuron_ld_max,
        )
        
    def become_dead(self):
        """ Transition neuron to dead state. Updates inflammation value, damage value, reach, and lipid droplets."""
        if self.dead or self.pos is None:
            return
        self.dead = True
        self.damaged = False
        self.healthy = False
        self.ticks_damaged = 0
        x, y = self.pos
        self.model.residence[x, y] = True
        self.model.res_curr[x, y] = max(self.model.res_curr[x, y], 1)
        # stronger pro-inflam & damage source, and no anti-inflam response from the neuron itself
        self.model.pro_inflam_val[x, y] = max(self.model.pro_inflam_val[x, y], 1.0)
        self.model.damage_val[x, y] = max(self.model.damage_val[x, y], 1.5)
        self.reach = 0
        self.lipid_droplets = min(
            self.lipid_droplets + self.model.neuron_ld_death_boost,
            self.model.neuron_ld_max,
        )

    def become_healthy(self):
        """ Transition neuron to healthy state. Resets damaged timer, updates reach and lipid droplets."""
        if self.dead or self.healthy:
            return
        self.damaged = False
        self.healthy = True
        self.ticks_damaged = 0
        # upon recovery, neuron can sense better and has fewer lipid droplets (e.g., via LD export or metabolism)
        self.reach = 2
        self.lipid_droplets *= 0.2 
        
    def _maybe_export_lipids_to_astrocytes(self):
        """  Export some lipid droplets to nearby astrocytes with some probability. This represents a potential protective mechanism where stressed neurons offload lipids to astrocytes for detoxification, but at the cost of boosting local inflammation and damage signals that can attract microglia. The balance of this process can influence the local environment and microglial responses."""
        if self.pos is None:
            return
        if self.lipid_droplets <= 0.0:
            return

        # find astrocytes in a radius
        x, y = self.pos
        neigh = self.model.grid.get_neighborhood(
            (x, y),
            moore=True,
            include_center=True,
            radius=self.model.neuron_to_astro_ld_transfer_radius,
        )
        cells = self.model.grid.get_cell_list_contents(neigh)
        astrocytes = [a for a in cells if isinstance(a, Astrocyte)]
        if not astrocytes:
            return

        # with some probability, export one "packet" of LD to a random astrocyte
        if self.model.random.random() < self.model.neuron_to_astro_ld_transfer_prob:
            target = self.model.random.choice(astrocytes)
            packet = min(
                self.lipid_droplets,
                self.model.neuron_ld_packet_size,
            )
            if packet > 0:
                self.lipid_droplets -= packet
                target.lipid_pool += packet


    def step(self):
        """ Main neuron behavior per tick:  
            - If healthy, can become damaged based on local pro/anti inflammatory cytokine ratio with some probability of becoming damaged.
            - If damaged, accumulates lipids and maybe offloads to nearby astrocyte, then can either stay damaged or transition:
                - become dead after tick threshold is met, pro/anti ratio is above threshold, and at some probability
                - becomes healthy again if pro/anti ratio is below threshold at some probability.
            - If dead, no changes.
            """
        if self.pos is None:
            return
        if self.dead:
            return
        
        # Lipid-driven damage pathway (separate from inflammation)
        if self.lipid_droplets >= 3.0 and not self.dead:
            if self.model.random.random() < 0.05:
                self.become_damaged()

        if self.damaged and (self.lipid_droplets >= 6.0 or self.toxic_lipids >= 5.0) and not self.dead:
            x, y = self.pos
            self.model.damage_val[x, y] += 0.5
        
        # Neuron lipid metabolism
        if self.lipid_droplets > 0:
            self.lipid_droplets *= 0.98  # 2% per tick decay
            
        if self.toxic_lipids > 0:
            self.toxic_lipids *= 0.95  # toxic species clear faster

        # --- params ---
        damage_ratio_thresh = self.model.damage_ratio_thresh
        healthy_ratio_thresh = self.model.healthy_ratio_thresh
        p_damage = self.model.damage_chance
        p_healthy = self.model.healthy_chance
        damage_to_death_ticks = self.model.damage_to_death_ticks
        death_ratio_thresh = self.model.death_ratio_thresh
        p_death = self.model.death_chance
        eps = 1e-9

        x, y = self.pos
        neigh_pos = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=False, radius=1
        )
        if not neigh_pos:
            return

        pro_vals = [self.model.pro_inflam_val[nx, ny] for (nx, ny) in neigh_pos]
        anti_vals = [self.model.anti_inflam_val[nx, ny] for (nx, ny) in neigh_pos]
        mean_pro = sum(pro_vals) / len(pro_vals)
        mean_anti = sum(anti_vals) / len(anti_vals)
        mean_ratio = mean_pro / (mean_anti + eps)

        # --- Healthy -> Damaged ---
        if self.healthy and (mean_ratio >= damage_ratio_thresh):
            if self.model.random.random() < p_damage:
                self.become_damaged()
            return

        # --- Damaged branch ---
        if self.damaged:
            self.ticks_damaged += 1
            
            self.lipid_droplets = min(
                self.lipid_droplets + self.model.neuron_ld_production_rate,
                self.model.neuron_ld_max,
            )
            
            self._maybe_export_lipids_to_astrocytes()

            # Damaged -> Dead
            if self.ticks_damaged >= damage_to_death_ticks:
                if mean_ratio >= death_ratio_thresh:
                    if self.model.random.random() < p_death:
                        self.become_dead()
                        return

            # Damaged -> Healthy
            if mean_ratio <= healthy_ratio_thresh:
                if self.model.random.random() < p_healthy:
                    self.become_healthy()
            return


class Astrocyte(Agent):
    """ Astrocyte agent: A0 ↔ A1 ↔ A2. """
    def __init__(self, unique_id, model, phenotype: str = "A0"):
        """Initialize astrocyte with state and parameters.

        Args:
            unique_id (int): unique identifier of agent
            model (Object): reference to the simulation model
            phenotype (str): initial phenotype of the astrocyte, one of "A0" (homeostatic), "A1" (pro-inflammatory), or "A2" (anti-inflammatory). Defaults to "A0".
            
        Internal State Variables:
            pos (Tuple[int, int]): current position of the astrocyte on the grid, used for sensing and interactions.
            wait_ticks (int): if the astrocyte is currently "busy" (e.g., performing a function), it will wait for this many ticks before acting again. Defaults to 0.
            resolution_ticks (int): counts how many ticks the astrocyte has been in a low-signal environment while in A1 or A2 phenotype, used to determine when to resolve back to A0. Defaults to 0.
            coverage_radius (int): how far the astrocyte can influence the local environment (e.g., cytokine modulation, neuron interactions). Defaults to model parameter.
            lipid_pool (float): represents the astrocyte's capacity to take up and detoxify lipids from neurons, which can influence its protective functions. Defaults to 1.0.
        """
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.phenotype = phenotype  # "A0", "A1", "A2"
        self.wait_ticks = 0
        self.resolution_ticks = 0  # how long in low-signal environment
        self.coverage_radius = self.model.astro_coverage_radius  # helps establish neighborhoods
        self.lipid_pool = 1.0
        
    def _oxidize_lipids(self):
        """ Oxidize lipids in the pool to produce anti-inflammatory signals, representing the astrocyte's role in detoxifying harmful lipid species from stressed neurons. This process is more active when there is a strong signal, and can help shift the local environment towards resolution. However, it also depletes the astrocyte's lipid pool, which can limit its protective capacity over time if not replenished."""
        if self.pos is None:
            return
        if self.lipid_pool <= 0.0:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        
        # If the local environment is below the astrocyte's sensing threshold, it won't activate the lipid oxidation process, representing a scenario where the astrocyte does not perceive enough stress or damage signals to trigger its protective functions. This allows the astrocyte to conserve its lipid pool and avoid unnecessary activation in a relatively healthy environment.
        if total < self.model.astro_homeo_signal_thresh:
            return
        
        # amount oxidized this tick
        frac = max(0.0, min(1.0, self.model.astro_ld_oxidation_rate))
        used = self.lipid_pool * frac
        if used <= 0.0:
            return
        
        #  oxidation produces anti-inflammatory signal
        self.lipid_pool -= used
        self.model.anti_inflam_val[x, y] += used * self.model.astro_ld_to_anti_inflam


    def _neighborhood_positions(self):
        """  Get positions in the neighborhood defined by coverage_radius. This is used for both sensing and modulating the local environment """
        x, y = self.pos
        neigh = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=True, radius=self.coverage_radius
        )
        return neigh

    def _nearby_microglia_counts(self):
        """Count nearby M1/M2 microglia in the astrocyte domain."""
        cells = self.model.grid.get_cell_list_contents(self._neighborhood_positions())
        num_M1 = 0
        num_M2 = 0
        for a in cells:
            # Microglia is defined later; this is resolved at runtime.
            if isinstance(a, Microglia):
                if a.phenotype == "M1":
                    num_M1 += 1
                elif a.phenotype == "M2":
                    num_M2 += 1
        return num_M1, num_M2

    def _update_phenotype(self):
        """ Astrocyte phnotype switching that considers two factors:
            - Microglia counts of a certain phenotype in the neighborhood, which biases astrocyte switching towards A1 then A2 with some probability 
            - Total inflammation and ratio of pro/anti inflammation:
                - If total signal is below a homeostatic threshold, allow drift back to A0.
                - If ratio is strongly pro, switch to A1 from A0 with some probability.
                - If ratio is strongly anti, switch to A2 from A0 with some probability.
                - If ratio is in the middle, allow drift back to A0 with some probability.
                - If currently A1 or A2 and total signal is low, count resolution ticks and switch back to A0 after threshold is met.
        """
        if self.pos is None:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        eps = 1e-9

        # Astrocyte-specific thresholds / probabilities
        homeo_thresh = self.model.astro_homeo_signal_thresh
        pro_thresh = self.model.astro_pro_inflam_signal_thresh
        anti_thresh = self.model.astro_anti_inflam_signal_thresh
        p_homeo = self.model.astro_homeo_chance
        p_pro = self.model.astro_pro_inflam_chance
        p_anti = self.model.astro_anti_inflam_chance

        # Microglia-driven induction bias
        num_M1, num_M2 = self._nearby_microglia_counts()
        r = self.model.random.random()

        if num_M1 > 0 and r < 0.3:
            # microglial A1-inducing cytokine environment
            self.phenotype = "A1"
        elif num_M2 > 0 and r < 0.3:
            # microglial A2-inducing environment
            self.phenotype = "A2"
        else:
            # fallback to local pro/anti ratio
            if total < homeo_thresh:
                # drift back to A0 over time
                if self.model.random.random() < p_homeo:
                    self.phenotype = "A0"
            else:
                ratio = pro / (anti + eps)
                r2 = self.model.random.random()

                # Strongly pro environment → A1 neurotoxic
                if ratio >= pro_thresh:
                    if r2 < p_pro:
                        self.phenotype = "A1"
                # Strongly anti / resolving environment → A2 protective
                elif ratio <= anti_thresh:
                    if r2 < p_anti:
                        self.phenotype = "A2"
                else:
                    # Balanced pro/anti ratio window: anti_thresh < ratio < pro_thresh.
                    if r2 < p_homeo:
                        self.phenotype = "A0"

        # resolution: A1/A2 -> A0 after long low-signal
        if self.phenotype in ("A1", "A2"):
            if total < homeo_thresh:
                self.resolution_ticks += 1
                if self.resolution_ticks >= self.model.a0_resolution_ticks:
                    self.phenotype = "A0"
                    self.resolution_ticks = 0
            else:
                self.resolution_ticks = 0
        else:
            self.resolution_ticks = 0

    def _modulate_fields(self):
        """ Astrocyte modulates local cytokine fields based on phenotype:
            - A1: increases pro-inflammatory signals in neighborhood, representing release of cytokines like IL-1β, TNFα, etc.
            - A2: increases anti-inflammatory signals and dampens pro-inflammatory signals, representing release of cytokines like IL-10, TGF-β, etc.
            - A0: mild cleanup, slightly reducing pro-inflammatory signals to represent homeostatic maintenance functions.
        """  
        for nx, ny in self._neighborhood_positions():
            if self.phenotype == "A1":
                # slightly stronger than before so it registers, but still < microglia
                self.model.pro_inflam_val[nx, ny] += 0.15
            elif self.phenotype == "A2":
                self.model.anti_inflam_val[nx, ny] += 0.15
                # damp the pro-inflam values in area
                self.model.pro_inflam_val[nx, ny] = max(
                    0.0, self.model.pro_inflam_val[nx, ny] - 0.07
                )
            else:
                # A0 homeostatic: mild pro cleanup (e.g., glutamate/ROS control)
                self.model.pro_inflam_val[nx, ny] *= 0.99

    def _neuron_in_radius(self):
        """ Get list of neurons within the astrocyte's coverage radius, used for direct interactions like glutamate clearance support (A2) or neurotoxicity (A1). """
        cells = self.model.grid.get_cell_list_contents(self._neighborhood_positions())
        return [a for a in cells if isinstance(a, Neuron)]

    def _act_on_neurons(self):
        """ Astrocyte acts on nearby neurons based on phenotype:
            - A2: strong local neuroprotection, helping damaged neurons recover with some probability, and slightly dismantling local damage/excitotoxicity to represent functions like glutamate clearance, antioxidant support, etc.
            - A1: can push healthy neurons into damage, or push damaged neurons to death, representing release of neurotoxic factors.
            - A0: small, background protection, representing homeostatic support functions.
        """
        neurons = self._neuron_in_radius()
        if not neurons:
            return

        for neuron in neurons:
            if self.phenotype == "A2":
                # strong local neuroprotection
                if neuron.damaged and not neuron.dead:
                    # base probability + small boost from lipid pool
                    base_p = 0.2
                    lipid_boost = min(self.lipid_pool * 0.01, 0.3)
                    p_heal = min(1.0, base_p + lipid_boost)
                    if self.model.random.random() < p_heal:
                        neuron.become_healthy()
                        
                # A2 also slightly dismantles local damage/excitotoxicity
                for nx, ny in self._neighborhood_positions():
                    self.model.damage_val[nx, ny] = max(
                        0.0, self.model.damage_val[nx, ny] - 0.1
                    )

            elif self.phenotype == "A1":
                # A1 can push healthy neurons into damage, or push damaged neurons to death
                if neuron.healthy:
                    if self.model.random.random() < 0.01:
                        neuron.become_damaged()
                elif neuron.damaged and not neuron.dead:
                    if self.model.random.random() < 0.05:
                        neuron.become_dead()
            else:
                # A0: small, background protection
                if neuron.damaged and not neuron.dead:
                    if self.model.random.random() < 0.02:
                        neuron.become_healthy()

    def _maybe_recruit_microglia(self):
        """  A1 astrocytes can recruit microglia if local damage and pro-inflammatory signals are high, representing the astrocyte's role in calling for help during injury or stress. This recruitment is probabilistic and depends on the local environment, ensuring that microglia are recruited when needed but not excessively in a healthy environment. Recruited microglia start as M1 pro-inflammatory phenotype, which can then further influence the local environment and astrocyte phenotypes."""
        if self.phenotype != "A1" or self.pos is None:
            return

        x, y = self.pos
        local_damage = float(self.model.damage_val[x, y])
        local_pro = float(self.model.pro_inflam_val[x, y])
        if (local_damage + local_pro) < self.model.recruitment_threshold:
            return

        # probability to recruit one new microglia agent (M1) which then gets sent to source
        if self.model.random.random() < self.model.microglia_recruitment_prob:
            m = Microglia(self.model.next_id(), self.model, phenotype="M1")
            self.model.grid.place_agent(m, (x, y))
            self.model.microglia.append(m)

    def step(self):
        """Main astrocyte behavior per tick:
            1) Update phenotype based on local microglia counts and cytokine ratios.
            2) Oxidize lipids to produce anti-inflammatory signals if there is a strong signal and the astrocyte has lipids in its pool.
            3) Modulate local cytokine fields based on phenotype.
            4) Act on nearby neurons based on phenotype (protection vs toxicity).
            5) If A1, maybe recruit microglia if local damage/pro signals are high."""
        if self.pos is None:
            return

        self._update_phenotype()
        self._oxidize_lipids()
        self._modulate_fields()
        self._act_on_neurons()
        self._maybe_recruit_microglia()


class Microglia(Agent):
    """
    Microglia with three phenotypes:
      M0 = homeostatic (surveillance)
      M1 = pro-inflammatory (attack)
      M2 = anti-inflammatory (repair)
    """

    def __init__(self, unique_id, model, phenotype: str, isLDAM: bool = False):
        """Initialize microglia with state and parameters.
        Args:
            unique_id (int): unique identifier of agent
            model (Object): reference to the simulation model
            phenotype (str): initial phenotype of the microglia, one of "M0" (homeostatic), "M1" (pro-inflammatory), or "M2" (anti-inflammatory).
        
        Internal State Variables:
            pos (Tuple[int, int]): current position of the microglia on the grid,
            isLDAM (bool): indicates whether the microglia is loaded with lipid droplets (LDAM).
            wait_ticks (int, optional): if the microglia is currently "busy" (e.g., performing phagocytosis or surveillance), it will wait for this many ticks before acting again. Defaults to 0.
            resolution_ticks (int, optional): counts how many ticks the microglia has been in a low-signal environment while in M1 or M2 phenotype, used to determine when to resolve back to M0. Defaults to 0.
            lipid_droplets (float, optional): represents the microglia's lipid droplet burden, which can influence its behavior and cytokine production. Defaults to 0.0.
            amyloid_exposure (float, optional): cumulative exposure to amyloid based on neighbors, which can influence activation and LD accumulation. Defaults to 0.0.
            amyloid_exposed_ticks (int, optional): how many ticks the microglia has been exposed to amyloid, which can influence chronic activation and dysfunction. Defaults to 0.
            eat_eff (float, optional): effective phagocytic efficiency, which can be influenced by phenotype and LD burden. Defaults to model base eat probability.
            sense_eff (float, optional): effective sensing efficiency, which can be influenced by phenotype and LD burden. Defaults to model base sensing efficiency.
            trem2_is_mutant (bool): whether or not microglia has a TREM2 mutation, which can influence its activation and function. Determined at initialization based on model trem2_mutation_rate.
            trem2_activity (float): activity level of TREM2, which is reduced if the microglia has a TREM2 mutation. This influences the strength of DAM activation in response to amyloid. Determined at initialization based on whether the microglia is mutant and model trem2_mutant_activity.
        """
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.phenotype = phenotype  # "M0", "M1", "M2"
        self.isLDAM = isLDAM
        self.isDAM = False
        self.dam_exposure_ticks = 0
        self.wait_ticks: int = 0
        self.resolution_ticks: int = 0  # how long M2 has sat in resolved environment
        self.lipid_droplets: float = 0.0 # LD burden in Microglia
        self.amyloid_exposure: float = 0.0 # cumulative exposure to amyloid based on neighbors, which can influence activation and LD accumulation
        self.amyloid_exposed_ticks: int = 0 # how many ticks the microglia has been exposed to amyloid, which can influence chronic activation and dysfunction
        self.eat_eff = model.base_eat_probability
        self.sense_eff = model.base_sensing_efficiency  
        # -------- TREM2 genotype (per-agent) --------
        self.trem2_is_mutant = (
            self.model.random.random() < self.model.trem2_mutation_rate
        )
        self.trem2_activity = (
            self.model.trem2_mutant_activity if self.trem2_is_mutant else 1.0
        )    
        
    def export_lipid_to_neurons(self):
        """ If this microglia is an LDAM, it can transfer some of its lipid droplets to nearby neurons, representing a potential toxic mechanism where dysfunctional microglia offload harmful lipid species to neurons, exacerbating neuronal stress and damage. This transfer is probabilistic and depends on the microglia's current lipid burden, ensuring that only microglia with significant LD accumulation contribute to this process. The transferred lipids can then influence the state of nearby neurons, potentially pushing them towards damage or death."""
        if not self.isLDAM or self.pos is None:
            return

        neigh = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=2
        )
        cells = self.model.grid.get_cell_list_contents(neigh)

        for agent in cells:
            if isinstance(agent, Neuron) and not agent.dead:
                self._transfer_toxic_lipid_to_neuron(agent)
                break

    def _transfer_toxic_lipid_to_neuron(self, neuron: "Neuron") -> bool:
        """Transfer one LDAM toxic-lipid packet to a living neuron."""
        if neuron.dead or self.lipid_droplets <= 0.0:
            return False
        transfer = min(
            self.model.ldam_neuron_toxic_lipid_transfer,
            self.lipid_droplets,
        )
        if transfer <= 0.0:
            return False
        self.lipid_droplets -= transfer
        neuron.toxic_lipids += transfer * self.model.apoe_multiplier
        return True
    
    def update_ldam_status(self):
        """ Update LDAM status based on lipid droplet burden, representing the idea that microglia with high lipid droplet accumulation can become dysfunctional and adopt characteristics such as impaired movement, phagocytosis, and increase in inflammatory cytokines and toxic lipid transfer to neurons"""
        if self.lipid_droplets >= self.model.ldam_threshold:
            self.isLDAM = True
        else:
            self.isLDAM = False
            
    def update_dam_status(self):
        """ Update DAM status based on local amyloid exposure and TREM2 activity, representing the process by which microglia sense amyloid in their environment and become activated into a DAM phenotype, which is influenced by TREM2 function. This activation can lead to increased phagocytic activity and cytokine production, but also contributes to chronic inflammation and potential neurotoxicity if sustained."""
        if self.pos is None:
            return

        # Measure local amyloid
        neigh = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )

        local_amyloid = sum(
            self._field_value(self.model.amyloid_val, p)
            for p in neigh
        )

        # TREM2-dependent DAM activation strength
        if local_amyloid >= self.model.dam_activation_threshold:
            self.dam_exposure_ticks += self.trem2_activity
        else:
            self.dam_exposure_ticks = max(0, self.dam_exposure_ticks - 1)
            

        if self.dam_exposure_ticks >= self.model.dam_activation_ticks:
            self.isDAM = True
        else:
            self.isDAM = False
            
    def compute_effective_params(self):
        """ Compute effective phagocytic and sensing efficiency based on phenotype and lipid droplet burden, representing how both the microglia's activation state and its lipid burden can influence its functional capabilities. This allows for a more dynamic representation of microglial behavior, where factors like DAM activation and LDAM dysfunction can modulate the microglia's ability to sense damage and clear debris."""
        eat = self.model.base_eat_probability
        sense = self.model.base_sensing_efficiency
        
        # Continuous lipid-dependent impairment. Recent amyloid exposure
        # amplifies the functional cost of accumulated lipid burden.
        lipid_impairment_scale = max(0.0, self.model.microglia_lipid_impairment_factor)
        if self.amyloid_exposed_ticks > 0:
            lipid_impairment_scale *= (
                1.0 + max(0.0, self.model.microglia_amyloid_impairment_factor)
            )
        lipid_factor = 1.0 / (1.0 + lipid_impairment_scale * self.lipid_droplets)
        eat *= lipid_factor
        sense *= lipid_factor

        # DAM increases activation
        if self.isDAM:
            eat *= 1.3
            sense *= 1.3

        # LDAM decreases function
        if self.isLDAM:
            eat *= 0.5
            sense *= 0.5

        self.eat_eff = min(1.0, eat)
        self.sense_eff = min(1.0, sense)

    def _update_phenotype(self):
        """
        Switch M0/M1/M2 based on a single ratio plus some random chance:

            ratio = pro / (anti + eps),

        plus:
        - a minimal total-signal threshold to allow homeostasis,
        - M1/M2 -> M0 if low-signal persists.
        """
        if self.pos is None:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        eps = 1e-9

        homeo_thresh = self.model.homeo_signal_thresh
        pro_thresh = self.model.pro_inflam_signal_thresh
        anti_thresh = self.model.anti_inflam_signal_thresh
        p_homeo = self.model.homeo_chance
        p_pro = self.model.pro_inflam_chance
        p_anti = self.model.anti_inflam_chance

        # Very low signal: allow drift back to M0
        if total < homeo_thresh:
            if self.model.random.random() < p_homeo:
                self.phenotype = "M0"
            # resolution logic handled below
        else:
            ratio = pro / (anti + eps)
            r = self.model.random.random()

            if ratio >= pro_thresh:
                if r < p_pro:
                    self.phenotype = "M1"
            elif ratio <= anti_thresh:
                if r < p_anti:
                    self.phenotype = "M2"
            else:
                # Balanced pro/anti ratio window: anti_thresh < ratio < pro_thresh.
                if r < p_homeo:
                    self.phenotype = "M0"

        # --- Resolution-based M1/M2 -> M0 ---
        if self.phenotype in ("M1", "M2"):
            if total < homeo_thresh:
                self.resolution_ticks += 1
                if self.resolution_ticks >= self.model.m0_resolution_ticks:
                    self.phenotype = "M0"
                    self.resolution_ticks = 0
            else:
                self.resolution_ticks = 0
        else:
            self.resolution_ticks = 0

    def _random_step(self):
        """ Random movement to a neighboring patch, used for surveillence when no strong chemotactic signal is detected. """
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if neighbors:
            new_pos = self.model.random.choice(neighbors)
            self.model.grid.move_agent(self, new_pos)

    def _chemotaxis_step(self, field_name: str):
        """
        Gradient-following that always moves one patch, using neuron_distance
        as the sensing radius for the damage/inflammation field.

        Used for chemotaxis up neuro-derived damage field.
        """
        field = getattr(self.model, field_name)
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if not neighbors:
            return

        sense_radius = max(1, int(self.model.neuron_distance))

        def sensed_field_value(pos):
            sensed_positions = self.model.grid.get_neighborhood(
                pos, moore=True, include_center=True, radius=sense_radius
            )
            return max(self._field_value(field, p) for p in sensed_positions)

        vals = [(n, sensed_field_value(n)) for n in neighbors]
        max_val = max(v for _, v in vals)
        candidates = [p for p, v in vals if v == max_val]
        new_pos = self.model.random.choice(candidates)
        self.model.grid.move_agent(self, new_pos)

    @staticmethod
    def _field_value(field, pos):
        """ Returns value of a field at a given position or zero if some error (e.g., out of bounds, or dict missing key) occurs. """
        try:
            return field[pos]
        except TypeError:
            # if field is a dict-like
            return field.get(pos, 0.0)

    # phenotype-specific movement
    def _move_M0(self):
        """Homeostatic: simple random surveillance."""
        self._random_step()

    def _move_M1(self):
        """Pro-inflammatory: chemotaxis up neuron-derived damage gradient."""
        if self.model.random.random() < self.sense_eff:
            self._chemotaxis_step("damage_val")
        else:
            self._random_step()

    def _move_M2(self):
        """Anti-inflammatory: same chemotaxis target (damage), different behavior at target."""
        if self.model.random.random() < self.sense_eff:
            self._chemotaxis_step("damage_val")
        else:
            self._random_step()

    # -------- interactions with neurons --------
    def _neurons_here(self):
        """All neuron agents at this patch."""
        cell_agents = self.model.grid.get_cell_list_contents([self.pos])
        return [a for a in cell_agents if isinstance(a, Neuron)]

    def _interact_M0(self) -> bool:
        """
        M0: surveillance + quiet clearance of dead neurons.

        - If a dead neuron is present and phagocytosed -> wait 5 ticks.
        - If *any* neuron is present but not eaten -> "surveillance" -> wait 5 ticks.
        - LDAM-specific: export toxic lipids to nearby neurons, representing a potential mechanism of neurotoxicity where dysfunctional microglia offload harmful lipid species to neurons, exacerbating neuronal stress and damage. This occurs even in the homeostatic phenotype if the microglia has become an LDAM, highlighting how lipid dysfunction can drive toxicity independent of traditional inflammatory activation states.
        """
        neurons = self._neurons_here()
        if not neurons:
            return False
        
        # Prefer to clear dead neurons if present
        for neuron in neurons:
            if neuron.dead and neuron.pos is not None:
                if self.model.random.random() < self.eat_eff:
                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 0.05
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
            # Export toxic lipids to living neurons only if LDAM
            elif (not neuron.dead) and neuron.pos is not None and self.isLDAM:
                self._transfer_toxic_lipid_to_neuron(neuron)

        # No dead neuron, but there is at least one neuron -> surveillance
        self.wait_ticks = 5
        return True

    def _interact_M1(self) -> bool:
        """
        M1: aggressive phagocytosis of damaged/dead neurons.

        - On successful phagocytosis -> wait 5 ticks.
        - If neuron(s) present but not phagocytosed -> "aggressive surveillance" -> wait 5 ticks.
        - LDAM-specific: export toxic lipids to nearby neurons, representing a potential mechanism of
        """
        neurons = self._neurons_here()
        for neuron in neurons:
            # First, try to phagocytose if damaged/dead
            if (neuron.damaged or neuron.dead) and neuron.pos is not None:
                p = min(1.0, self.eat_eff * 1.2)
                if self.model.random.random() < p:
                    ld_from_neuron = getattr(neuron, "lipid_droplets", 0.0)
                    self.lipid_droplets += ld_from_neuron + self.model.microglia_ld_from_phagocytosis
                    self.lipid_droplets = min(self.lipid_droplets, self.model.microglia_ld_max)

                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.pro_inflam_val[x, y] += 1.0
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
            # After phagocytosis check, export toxic lipids if LDAM and neuron still exists and is not dead
            elif (not neuron.dead) and self.isLDAM and neuron.pos is not None:
                # Export toxic lipids to Neurons from lipid pool (only if still alive)
                self._transfer_toxic_lipid_to_neuron(neuron)
        return False

    def _interact_M2(self) -> bool:
        """
        M2: repair/cleanup.
          - clears dead neurons
          - helps damaged neurons recover

        - Any successful phagocytosis or healing -> wait 5 ticks.
        - If neuron(s) present but not phagocytosed/healed -> "repair surveillance" -> wait 5 ticks.
        - LDAM-specific: export toxic lipids to nearby neurons, representing a potential mechanism of
        """
        neurons = self._neurons_here()
        for neuron in neurons:
            # Check for dead neurons first - phagocytose them
            if neuron.dead and neuron.pos is not None:
                if self.model.random.random() < self.eat_eff:
                    # capture neuron LD + baseline LD from phagocytosis
                    ld_from_neuron = getattr(neuron, "lipid_droplets", 0.0)
                    self.lipid_droplets += ld_from_neuron + self.model.microglia_ld_from_phagocytosis
                    self.lipid_droplets = min(self.lipid_droplets, self.model.microglia_ld_max)
                    
                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 1.0
                    # dismantle damage wave from this site
                    self.model.dismantling[x, y] = True
                    self.model.dis_curr[x, y] = self.model.damage_radius
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
            # Check for damaged neurons - help them recover
            elif neuron.damaged and (not neuron.dead) and neuron.pos is not None:
                if self.model.random.random() < self.sense_eff:
                    neuron.become_healthy()
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 1.0
                    self.model.pro_inflam_val[x, y] = max(
                        0.0, self.model.pro_inflam_val[x, y] - 1.0
                    )
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
            # Export toxic lipids to living neurons only at the end of processing
            elif (not neuron.dead) and self.isLDAM and neuron.pos is not None:
                # Export toxic lipids to living neurons from lipid pool
                self._transfer_toxic_lipid_to_neuron(neuron)
        return False


    # -------- cytokine emission along path --------
    def _emit_cytokines_at(self, pos):
        """
        Emit cytokines at (and around) the given position.
        Called after *each move*, so microglia leave a trail along the damage gradient depending on phenotype.
        LDAM burden can boost pro-inflammatory emission even for M1 microglia, representing the idea that LDAM microglia are more pro-inflammatory and can create a more hostile environment that further attracts microglia and damages neurons. This creates a positive feedback loop where stressed microglia with high LDs amplify inflammation in their vicinity.
        """
        if pos is None:
            return

        # local neighborhood
        neigh = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        
        # Lipid-droplet-boosted pro-inflammatory trail, if LD burden is above threshold, boost pro-inflammatory signal emitted along path, representing idea that LD-burdened microglia are more pro-inflammatory and can create a more hostile environment that further attracts microglia and damages neurons. This creates a positive feedback loop where stressed microglia with high LDs amplify inflammation in their vicinity.
        lipid_pro_inflam_active = (
            self.lipid_droplets >= self.model.microglia_ld_pro_inflam_threshold
        )
        ldam_factor = 1.0
        if lipid_pro_inflam_active:
            ldam_factor += self.model.microglia_ldam_inflam_boost

        if self.phenotype == "M1":
            # pro-inflam trail
            for nx, ny in neigh:
                self.model.pro_inflam_val[nx, ny] += 0.2 * ldam_factor  # small per-move increment
        elif self.phenotype == "M2":
            # anti-inflam trail
            for nx, ny in neigh:
                self.model.anti_inflam_val[nx, ny] += 0.2
                if lipid_pro_inflam_active:
                    self.model.pro_inflam_val[nx,ny] += 0.2 * ldam_factor
                
    # instead of repeating the same code in each interaction method, we can just call this after interaction
    def amyloid_exposure_update(self):
        """ Update amyloid exposure and resulting effects, representing the microglia's role in sensing and responding to amyloid pathology. This method checks the local amyloid environment, updates the microglia's cumulative exposure, and applies consequences such as lipid droplet accumulation, amyloid clearance, and increased pro-inflammatory signaling. Functional impairment is recalculated centrally in compute_effective_params()."""
        # Check if amyloid reaches fibril/plaque levels; if so, add lipid
        # burden, clear nearby amyloid, and emit a pro-inflammatory signal.
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        self.amyloid_exposure = sum(
            self._field_value(self.model.amyloid_val, neighbor)
            for neighbor in neighborhood
        )
        max_exposure = max(
            self._field_value(self.model.amyloid_val, neighbor)
            for neighbor in neighborhood
        )
        if self.amyloid_exposure >= self.model.amyloid_fibril_thresh or max_exposure >= self.model.amyloid_fibril_thresh:
            self.lipid_droplets += (
                self.model.microglia_ld_from_amyloid
                * self.trem2_activity
                * self.model.apoe_multiplier
            )
            self.lipid_droplets = min(self.lipid_droplets, self.model.microglia_ld_max)
            self.amyloid_exposed_ticks += 1
            for neighbor in neighborhood:
                if self.model.amyloid_val[neighbor] > self.model.amyloid_fibril_thresh:
                    self.model.amyloid_val[neighbor] = max(
                        0.0,
                        self.model.amyloid_val[neighbor]
                        - self.model.microglia_amyloid_clearance * self.trem2_activity
                    )
            x, y = self.pos
            self.model.pro_inflam_val[x, y] += 0.5
        else:
            self.amyloid_exposed_ticks = max(0, self.amyloid_exposed_ticks - 1)

    def step(self):
        """ Step logic:
            - Update disease states (DAM/LDAM) based on current environment and internal state.
            - Update effective parameters (phagocytosis/sensing) based on phenotype
            - If "busy" (engulfing / surveilling), just count down.
            - If not busy, update phenotype based on cytokine ratio + resolution.
            - Move with phenotype-specific speed and movement logic.
            - After moving, interact with neurons at new location based on phenotype.
            - Emit cytokines along the path after each move, with potential LD-boosted pro-inflammatory emission for M1 microglia with high LD burden.
            - Update amyloid exposure and resulting effects after interactions, representing the microglia's role in sensing and responding to amyloid pathology. This allows the model to capture how microglia can become activated and dysfunctional in response to amyloid, which is a key aspect of Alzheimer's disease pathology.
            - Export toxic lipids to neurons if LDAM after interactions, representing a potential mechanism of neurotoxicity where dysfunctional microglia offload harmful lipid species to neurons, exacerbating neuronal stress and damage. This occurs even in the homeostatic phenotype if the microglia has become an LDAM, highlighting how lipid dysfunction can drive toxicity independent of traditional inflammatory activation states.
        """
        
        # If "busy" (engulfing / surveilling), just count down
        if self.pos is None:
            return
        
        # 1) update disease states
        self.update_ldam_status()
        
        self.update_dam_status()
        
        if self.lipid_droplets > 0:
            burn = self.lipid_droplets * self.model.microglia_ld_oxidation_rate
            self.lipid_droplets = max(0.0, self.lipid_droplets - burn)

            # phenotype-specific signaling from oxidation
            if self.phenotype == "M1":
                x, y = self.pos
                self.model.pro_inflam_val[x, y] += burn * self.model.microglia_ld_to_pro_inflam
            else:
                x, y = self.pos
                self.model.anti_inflam_val[x, y] += burn * self.model.microglia_ld_to_anti_inflam

        self.update_ldam_status()

        # 2️) Compute effective parameters for THIS tick
        self.compute_effective_params()
        
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        # 3️) Update phenotype (M0/M1/M2)
        self._update_phenotype()

        # 4) Movement with phenotype-specific speed
        base_speed = max(0.0, 1.0 + 0.05 * (72 - 37.0))  # your original temperature logic
        if self.phenotype == "M1":
            speed_factor = 1.5
        else:
            speed_factor = 1.0
        ldam_speed_factor = 0.5 if self.isLDAM else 1.0  # LDAM microglia are slower due to dysfunction
        speed_factor *= ldam_speed_factor
        speed = base_speed * speed_factor
        moves = 1
        extra = speed - 1.0
        extra_move_prob = max(0.0, min(1.0, extra))
        if self.isLDAM and extra > 1.0:
            # The two-move cap can otherwise hide LDAM slowing for fast M1 cells.
            extra_move_prob *= ldam_speed_factor
        if self.model.random.random() < extra_move_prob:
            moves += 1

        acted = False
        for _ in range(moves):
            if self.phenotype == "M1":
                self._move_M1()
                acted = self._interact_M1()
            elif self.phenotype == "M2":
                self._move_M2()
                acted = self._interact_M2()
            else:  # M0
                self._move_M0()
                acted = self._interact_M0()

            # emit cytokines along the path
            self._emit_cytokines_at(self.pos)
            # update amyloid exposure and potential consequences
            self.amyloid_exposure_update()
            
            self.export_lipid_to_neurons()

            if acted:
                break


class MicrogliaNeuronModel(Model):
    """ Model class that holds the grid and manages agents. Initializes with parameters for grid size, seeding, microglia behavior, neuron behavior, astrocyte behavior, signal decay, resolution times, and lipid dynamics. """
    def __init__(
        self,
        width: int = 33,
        height: int = 33,
        torus: bool = True,
        # microglia seeding
        init_m0_microglia: int = 5,
        init_m1_microglia: int = 5,
        init_m2_microglia: int = 5,
        # neuron seeding
        init_healthy_neuron: int = 10,
        init_damaged_neuron: int = 10,
        init_dead_neuron: int = 10,
        # astrocyte seeding (by phenotype)
        init_a0_astrocytes: int = 10,
        init_a1_astrocytes: int = 5,
        init_a2_astrocytes: int = 5,
        astro_coverage_radius: int = 2,
        # core microglia behavior
        base_eat_probability: float | None = None,  # defaults to eat_probability
        eat_probability: float = 0.70,
        sensing_efficiency: float = 0.50,
        base_sensing_efficiency: float | None = None,  # defaults to sensing_efficiency
        damage_chance: float = 0.005,
        neuron_distance: int = 5,
        damage_radius: int = 3,
        seed: int | None = None,
        # Neuron knobs
        damage_ratio_thresh: float = 0.6,
        healthy_ratio_thresh: float = 0.2,
        healthy_chance: float = 0.20,
        damage_to_death_ticks: int = 20,
        death_ratio_thresh: float = 0.9,
        death_chance: float = 0.25,
        # Microglia / Astrocyte phenotype signal thresholds (microglia)
        homeo_signal_thresh: float = 1.0,
        pro_inflam_signal_thresh: float = 1.05,
        anti_inflam_signal_thresh: float = 0.95,
        homeo_chance: float = 0.01,
        pro_inflam_chance: float = 0.2,
        anti_inflam_chance: float = 0.2,
        # Astrocyte-specific phenotype thresholds
        astro_homeo_signal_thresh: float = 0.5,
        astro_pro_inflam_signal_thresh: float = 1.05,
        astro_anti_inflam_signal_thresh: float = 0.95,
        astro_homeo_chance: float = 0.01,
        astro_pro_inflam_chance: float = 0.3,
        astro_anti_inflam_chance: float = 0.3,
        # signal decay
        pro_decay: float = 0.20,
        anti_decay: float = 0.10,
        # resolution times (separate for microglia vs astrocytes)
        m0_resolution_ticks: int = 10,         # microglia M1/M2 -> M0
        a0_resolution_ticks: int = 10,         # astrocyte A1/A2 -> A0
        # astrocyte-driven microglia recruitment (optional; used inside Astrocyte)
        recruitment_threshold: float = 3.0,
        microglia_recruitment_prob: float = 0.01,
        # ---- lipid dynamics ----
        neuron_ld_production_rate: float = 0.2,
        neuron_ld_damage_boost: float = 0.5,
        neuron_ld_death_boost: float = 1.0,
        neuron_ld_packet_size: float = 0.5,
        neuron_to_astro_ld_transfer_prob: float = 0.3,
        neuron_to_astro_ld_transfer_radius: int = 2,
        astro_ld_oxidation_rate: float = 0.05,
        astro_ld_to_anti_inflam: float = 0.05,
        microglia_ld_from_phagocytosis: float = 0.5,
        microglia_ld_pro_inflam_threshold: float = 5.0,
        # ldam 
        microglia_ldam_inflam_boost: float = 0.5,
        ldam_neuron_toxic_lipid_transfer: float = 0.1,
        ldam_threshold: float = 5.0,
        
        # dam 
        dam_activation_threshold: float = 3.0,
        dam_activation_ticks: int = 5,
        
        # amyloid related arguments
        initial_amyloid_patches: int = 5,
        amyloid_diffusion: float = 0.1,
        amyloid_decay: float = 0.05,
        amyloid_growth: float = 0.1,
        amyloid_carrying_capacity: float = 5.0,
        amyloid_fibril_thresh: float = 2.0,
        microglia_ld_from_amyloid: float = 1.0,
        microglia_amyloid_clearance: float = 0.5,
        microglia_lipid_impairment_factor: float = 0.05,
        microglia_amyloid_impairment_factor: float = 0.2,
        
        # --- Genotype parameters ---
        apoe_genotype: str = "3/3",              # "3/3", "3/4", "4/4"
        trem2_mutation_rate: float = 0.25,       # probability microglia are R47H
        trem2_mutant_activity: float = 0.5,      # loss-of-function level (0-1)
    ):
        """Model initialization with parameters for grid, agent seeding, behavior probabilities, signal decay, resolution times, and lipid dynamics.

        Args:
            width (int, optional): Width of the grid. Defaults to 33.
            height (int, optional): Height of the grid. Defaults to 33.
            torus (bool, optional): Whether the grid is toroidal. Defaults to True.
            init_m0_microglia (int, optional): Initial number of M0 microglia. Defaults to 5.
            init_m1_microglia (int, optional): Initial number of M1 microglia. Defaults to 5.
            init_m2_microglia (int, optional): Initial number of M2 microglia. Defaults to 5.
            init_healthy_neuron (int, optional): Initial number of healthy neurons. Defaults to 10.
            init_damaged_neuron (int, optional): Initial number of damaged neurons. Defaults to 10.
            init_dead_neuron (int, optional): Initial number of dead neurons. Defaults to 10.
            init_a0_astrocytes (int, optional): Initial number of A0 astrocytes. Defaults to 10.
            init_a1_astrocytes (int, optional): Initial number of A1 astrocytes. Defaults to 5.
            init_a2_astrocytes (int, optional): Initial number of A2 astrocytes. Defaults to 5.
            astro_coverage_radius (int, optional): Astrocyte coverage radius. Defaults to 2.
            eat_probability (float, optional): Probability that a microglia will eat a neuron. Defaults to 0.70.
            base_eat_probability (float, optional): Optional baseline used when recalculating effective phagocytosis after impairment. Defaults to eat_probability.
            sensing_efficiency (float, optional): Baseline efficiency of microglia sensing signals. Defaults to 0.50.
            base_sensing_efficiency (float, optional): Optional baseline used when recalculating effective sensing after impairment. Defaults to sensing_efficiency.
            damage_chance (float, optional): Chance that a neuron will be damaged. Defaults to 0.005.
            neuron_distance (int, optional): Distance between neurons in the grid. Defaults to 5.
            damage_radius (int, optional): Radius within which a damaged neuron affects nearby cells. Defaults to 3.
            seed (int | None, optional): Simulation seed. Defaults to None.
            damage_ratio_thresh (float, optional): Damage ratio threshold for triggering certain behaviors. Defaults to 0.6.
            healthy_ratio_thresh (float, optional): Healthy neuron ratio threshold for triggering certain behaviors. Defaults to 0.2.
            healthy_chance (float, optional): Chance that a neuron will be healthy. Defaults to 0.20.
            damage_to_death_ticks (int, optional): Number of ticks after which a damaged neuron dies. Defaults to 20.
            death_ratio_thresh (float, optional): Death ratio threshold for triggering certain behaviors. Defaults to 0.9.
            death_chance (float, optional): Chance that a damaged neuron will die. Defaults to 0.25.
            homeo_signal_thresh (float, optional): Homeostatic signal threshold for triggering certain behaviors. Defaults to 1.0.
            pro_inflam_signal_thresh (float, optional): Pro/anti ratio threshold above which microglia are biased toward M1. Defaults to 1.05.
            anti_inflam_signal_thresh (float, optional): Pro/anti ratio threshold below which microglia are biased toward M2. Defaults to 0.95.
            homeo_chance (float, optional): Chance that a microglia will produce homeostatic signals. Defaults to 0.01.
            pro_inflam_chance (float, optional): Chance that a microglia will produce pro-inflammatory signals. Defaults to 0.2.
            anti_inflam_chance (float, optional): Chance that a microglia will produce anti-inflammatory signals. Defaults to 0.2.
            astro_homeo_signal_thresh (float, optional): Astrocyte homeostatic signal threshold for triggering certain behaviors. Defaults to 0.5.
            astro_pro_inflam_signal_thresh (float, optional): Pro/anti ratio threshold above which astrocytes are biased toward A1. Defaults to 1.05.
            astro_anti_inflam_signal_thresh (float, optional): Pro/anti ratio threshold below which astrocytes are biased toward A2. Defaults to 0.95.
            astro_homeo_chance (float, optional): Chance that an astrocyte will produce homeostatic signals. Defaults to 0.01.
            astro_pro_inflam_chance (float, optional): Chance that an astrocyte will produce pro-inflammatory signals. Defaults to 0.3.
            astro_anti_inflam_chance (float, optional): Chance that an astrocyte will produce anti-inflammatory signals. Defaults to 0.3.
            pro_decay (float, optional): Pro-inflammatory signal decay rate. Defaults to 0.20.
            anti_decay (float, optional): Anti-inflammatory signal decay rate. Defaults to 0.10.
            m0_resolution_ticks (int, optional): Number of low-signal ticks to resolve M1 or M2 to M0 microglia transition. Defaults to 10.
            a0_resolution_ticks (int, optional): Number of low-signal ticks to resolve A1 or A2 to A0 astrocyte transition. Defaults to 10.
            microglia_recruitment_prob (float, optional): Probability that a microglia will be recruited. Defaults to 0.01.
            neuron_ld_production_rate (float, optional): Rate at which neurons produce lipid droplets. Defaults to 0.2.
            neuron_ld_damage_boost (float, optional): Boost in lipid droplet production due to damage. Defaults to 0.5.
            neuron_ld_death_boost (float, optional): Boost in lipid droplet production due to neuron death. Defaults to 1.0.
            neuron_ld_packet_size (float, optional): Size of lipid droplet packets. Defaults to 0.5.
            neuron_to_astro_ld_transfer_prob (float, optional): Probability that a neuron will transfer lipid droplets to an astrocyte. Defaults to 0.3.
            neuron_to_astro_ld_transfer_radius (int, optional): Radius within which neurons can transfer lipid droplets to astrocytes. Defaults to 2.
            astro_ld_oxidation_rate (float, optional): Rate at which astrocyte lipid droplets are oxidized. Defaults to 0.05.
            astro_ld_to_anti_inflam (float, optional): Rate at which astrocyte lipid droplets are converted to anti-inflammatory signals. Defaults to 0.05.
            microglia_ld_from_phagocytosis (float, optional): Lipid droplet production from microglia phagocytosis. Defaults to 0.5.
            microglia_ld_pro_inflam_threshold (float, optional): Threshold for microglia lipid droplet production of pro-inflammatory signals. Defaults to 5.0.
            microglia_ldam_inflam_boost (float, optional): Boost in microglia lipid droplet production of pro-inflammatory signals when threshold is met. Defaults to 0.5.
            ldam_neuron_toxic_lipid_transfer (float, optional): Amount of toxic lipids transferred from LDAM microglia to neurons per tick. Defaults to 0.1.
            ldam_threshold (float, optional): Threshold for LDAM microglia activation. Defaults to 5.0.
            dam_activation_threshold (float, optional): Threshold for DAM activation. Defaults to 3.0.
            dam_activation_ticks (int, optional): Number of ticks for DAM activation. Defaults to 5.
            initial_amyloid_patches (int, optional): Initial number of amyloid patches. Defaults to 5.
            amyloid_diffusion (float, optional): Amyloid diffusion rate. Defaults to 0.1.
            amyloid_decay (float, optional): Amyloid decay rate. Defaults to 0.05.
            amyloid_growth (float, optional): Amyloid logistic growth rate. Defaults to 0.1.
            amyloid_carrying_capacity (float, optional): Per-patch carrying capacity K used to cap logistic amyloid growth. Defaults to 5.0.
            amyloid_fibril_thresh (float, optional): Amyloid threshold for fibril formation. Defaults to 2.0.
            microglia_ld_from_amyloid (float, optional): Lipid droplet production from amyloid exposure. Defaults to 1.0.
            microglia_amyloid_clearance (float, optional): Rate at which microglia clear amyloid. Defaults to 0.5.
            microglia_lipid_impairment_factor (float, optional): Baseline strength of lipid-burden impairment on microglial sensing and phagocytosis. Defaults to 0.05.
            microglia_amyloid_impairment_factor (float, optional): Extra multiplier on lipid-burden impairment while amyloid exposure persists. Defaults to 0.2.
            apoe_genotype (str, optional): APOE genotype of the model. Defaults to "3/3".
            trem2_mutation_rate (float, optional): Probability that microglia have a TREM2 mutation. Defaults to 0.25.
            trem2_mutant_activity (float, optional): Activity level of TREM2 mutant microglia. Defaults to 0.5.
        """
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=torus)
        self.width, self.height = width, height

        # counters
        self._uid: int = 0
        self.steps: int = 0

        # params
        self.eat_probability = float(eat_probability)
        if base_eat_probability is None:
            base_eat_probability = eat_probability
        if base_sensing_efficiency is None:
            base_sensing_efficiency = sensing_efficiency
        self.sensing_efficiency = float(sensing_efficiency)
        self.base_eat_probability = float(base_eat_probability)
        self.base_sensing_efficiency = float(base_sensing_efficiency)
        self.damage_chance = float(damage_chance)
        self.neuron_distance = int(neuron_distance)
        self.damage_radius = int(damage_radius)

        # neuron params
        self.damage_ratio_thresh = float(damage_ratio_thresh)
        self.healthy_ratio_thresh = float(healthy_ratio_thresh)
        self.healthy_chance = float(healthy_chance)
        self.damage_to_death_ticks = int(damage_to_death_ticks)
        self.death_ratio_thresh = float(death_ratio_thresh)
        self.death_chance = float(death_chance)

        # microglia phenotype signal params
        self.homeo_signal_thresh = float(homeo_signal_thresh)
        self.pro_inflam_signal_thresh = float(pro_inflam_signal_thresh)
        self.anti_inflam_signal_thresh = float(anti_inflam_signal_thresh)
        self.homeo_chance = float(homeo_chance)
        self.pro_inflam_chance = float(pro_inflam_chance)
        self.anti_inflam_chance = float(anti_inflam_chance)

        # astrocyte phenotype signal params
        self.astro_homeo_signal_thresh = float(astro_homeo_signal_thresh)
        self.astro_pro_inflam_signal_thresh = float(astro_pro_inflam_signal_thresh)
        self.astro_anti_inflam_signal_thresh = float(astro_anti_inflam_signal_thresh)
        self.astro_homeo_chance = float(astro_homeo_chance)
        self.astro_pro_inflam_chance = float(astro_pro_inflam_chance)
        self.astro_anti_inflam_chance = float(astro_anti_inflam_chance)

        # signal decay params
        self.pro_decay = float(pro_decay)
        self.anti_decay = float(anti_decay)

        # resolution times
        self.m0_resolution_ticks = int(m0_resolution_ticks)  # microglia
        self.a0_resolution_ticks = int(a0_resolution_ticks)  # astrocytes

        # astrocyte params
        self.astro_coverage_radius = int(astro_coverage_radius)
        self.recruitment_threshold = float(recruitment_threshold)
        self.microglia_recruitment_prob = float(microglia_recruitment_prob)
        
        # lipid dynamics params
        self.neuron_ld_production_rate = float(neuron_ld_production_rate)
        self.neuron_ld_damage_boost = float(neuron_ld_damage_boost)
        self.neuron_ld_death_boost = float(neuron_ld_death_boost)
        self.neuron_ld_packet_size = float(neuron_ld_packet_size)
        self.neuron_to_astro_ld_transfer_prob = float(neuron_to_astro_ld_transfer_prob)
        self.neuron_to_astro_ld_transfer_radius = int(neuron_to_astro_ld_transfer_radius)

        self.astro_ld_oxidation_rate = float(astro_ld_oxidation_rate)
        self.astro_ld_to_anti_inflam = float(astro_ld_to_anti_inflam)

        self.microglia_ld_from_phagocytosis = float(microglia_ld_from_phagocytosis)
        self.microglia_ld_pro_inflam_threshold = float(microglia_ld_pro_inflam_threshold)
        self.microglia_ld_oxidation_rate: float = 0.02   # fraction burned per tick
        self.microglia_ld_to_pro_inflam: float = 0.05    # pro signal per LD used (M1)
        self.microglia_ld_to_anti_inflam: float = 0.05   # anti signal per LD used (M0/M2
        self.microglia_ld_max: float = 50.0 
        self.neuron_ld_max: float = 10.0
        
        # ldam
        self.microglia_ldam_inflam_boost = float(microglia_ldam_inflam_boost)
        self.ldam_neuron_toxic_lipid_transfer = float(ldam_neuron_toxic_lipid_transfer)  # amount of toxic lipids transferred from LDAM microglia to neurons per tick
        self.ldam_threshold = float(ldam_threshold)  # threshold for LDAM microglia activation
        
        # dam
        self.dam_activation_threshold = float(dam_activation_threshold)
        self.dam_activation_ticks = int(dam_activation_ticks)
        
        # -------- Genotype --------
        self.apoe_genotype = apoe_genotype
        self.trem2_mutation_rate = trem2_mutation_rate
        self.trem2_mutant_activity = trem2_mutant_activity

        # APOE genotype multiplier (lipid bias)
        apoe_map = {"3/3": 1.0, "3/4": 1.5, "4/4": 2.5}
        self.apoe_multiplier = apoe_map.get(apoe_genotype, 1.0)

        # patch fields
        shp = (width, height)
        # neuron-sourced damage field for chemotaxis
        self.damage_val = np.zeros(shp, dtype=float)
        # cytokine fields
        self.pro_inflam_val = np.zeros(shp, dtype=float)
        self.anti_inflam_val = np.zeros(shp, dtype=float)

        self.residence = np.zeros(shp, dtype=bool)
        self.res_curr = np.zeros(shp, dtype=int)
        self.dismantling = np.zeros(shp, dtype=bool)
        self.dis_curr = np.zeros(shp, dtype=int)
        
        # patch amyloid plaque fields (concentration with thresholds represent monomers, polymers, plaques with different sizes and movement dynamics)
        self.amyloid_val = np.zeros(shp, dtype=float)
        self.amyloid_decay = float(amyloid_decay)  # decay rate of amyloid concentration per tick
        self.amyloid_growth = float(amyloid_growth)   # logistic growth rate of amyloid concentration per tick
        self.amyloid_carrying_capacity = float(amyloid_carrying_capacity)
        self.diffusion_rate = float(amyloid_diffusion)     # diffusion rate of amyloid across the grid
        self.init_amyloid_patches = int(initial_amyloid_patches)
        
        # microglia amyloid interaction params
        self.amyloid_fibril_thresh = float(amyloid_fibril_thresh)
        self.microglia_ld_from_amyloid = float(microglia_ld_from_amyloid)
        self.microglia_amyloid_clearance = float(microglia_amyloid_clearance)
        self.microglia_lipid_impairment_factor = float(microglia_lipid_impairment_factor)
        self.microglia_amyloid_impairment_factor = float(microglia_amyloid_impairment_factor)

        # agents
        self.microglia: list[Microglia] = []
        self.neurons: list[Neuron] = []
        self.astrocytes: list[Astrocyte] = []

        # -------- neurons initialized --------
        for _ in range(init_healthy_neuron):
            n = Neuron(self.next_id(), self, damaged=False, dead=False)
            self._place_random(n)
            self.neurons.append(n)

        for _ in range(init_damaged_neuron):
            n = Neuron(self.next_id(), self, damaged=True, dead=False)
            self._place_random(n)
            self.neurons.append(n)
            x, y = n.pos
            self.residence[x, y] = True
            self.res_curr[x, y] = 1
            self.pro_inflam_val[x, y] = max(self.pro_inflam_val[x, y], 1.0)
            self.damage_val[x, y] = max(self.damage_val[x, y], 1.0)

        for _ in range(init_dead_neuron):
            n = Neuron(self.next_id(), self, damaged=False, dead=True)
            self._place_random(n)
            self.neurons.append(n)
            x, y = n.pos
            self.residence[x, y] = True
            self.res_curr[x, y] = 1
            self.pro_inflam_val[x, y] = max(self.pro_inflam_val[x, y], 1.0)
            self.damage_val[x, y] = max(self.damage_val[x, y], 1.5)

        # -------- microglia initialized --------
        for _ in range(init_m0_microglia):
            m = Microglia(self.next_id(), self, phenotype="M0")
            self._place_random(m)
            self.microglia.append(m)
        for _ in range(init_m1_microglia):
            m = Microglia(self.next_id(), self, phenotype="M1")
            self._place_random(m)
            self.microglia.append(m)
        for _ in range(init_m2_microglia):
            m = Microglia(self.next_id(), self, phenotype="M2")
            self._place_random(m)
            self.microglia.append(m)

        # -------- astrocytes initialized --------
        for _ in range(init_a0_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A0",
            )
            self._place_random(a)
            self.astrocytes.append(a)

        for _ in range(init_a1_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A1",
            )
            self._place_random(a)
            self.astrocytes.append(a)

        for _ in range(init_a2_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A2",
            )
            self._place_random(a)
            self.astrocytes.append(a)
            
        # Initialize level of amyloid randomly in the grid, with some patches having higher concentrations to represent plaques
        for _ in range(self.init_amyloid_patches):  # number of initial plaques
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.amyloid_val[x, y] = self.random.uniform(0, 5)  # random initial concentration for plaques

        # -------- DataCollector --------
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.steps,
                # neurons
                "damaged_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.damaged)
                ),
                "dead_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.dead)
                ),
                "healthy_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.healthy)
                ),
                "neurons_total": lambda m: sum(
                    1 for n in m.neurons if n.pos is not None
                ),
                # cytokine fields
                "total_pro_inflammation": lambda m: float(m.pro_inflam_val.sum()),
                "total_anti_inflammation": lambda m: float(m.anti_inflam_val.sum()),
                "mean_pro_inflammation": lambda m: float(m.pro_inflam_val.mean()),
                "mean_anti_inflammation": lambda m: float(m.anti_inflam_val.mean()),
                # microglia
                "microglia_total": lambda m: len(m.microglia),
                "microglia_M0": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M0"
                ),
                "microglia_M1": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M1"
                ),
                "microglia_M2": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M2"
                ),
                "microglia_LDAM": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.isLDAM
                ),
                "microglia_DAM": lambda m: sum(
                    1 for g in m.microglia                    
                    if g.pos is not None and g.isDAM
                ),
                # astrocytes
                "astrocytes_total": lambda m: len(m.astrocytes),
                "astrocytes_A0": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A0"
                ),
                "astrocytes_A1": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A1"
                ),
                "astrocytes_A2": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A2"
                ),
                # lipid droplet metrics
                "total_ld_neurons": lambda m: float(sum(
                    getattr(n, "lipid_droplets", 0.0)
                    for n in m.neurons
                    if n.pos is not None
                )),
                "total_ld_microglia": lambda m: float(sum(
                    getattr(g, "lipid_droplets", 0.0)
                    for g in m.microglia
                    if g.pos is not None
                )),
                "total_ld_astrocytes": lambda m: float(sum(
                    getattr(a, "lipid_pool", 0.0)
                    for a in m.astrocytes
                    if a.pos is not None
                )),
                # Track amyloid concentration totals and occupied patch counts.
                "total_amyloid_value": lambda m: float(m.amyloid_val.sum()),
                "total_amyloid_patches": lambda m: int(np.count_nonzero(m.amyloid_val > 0.0)),
                "amyloid_positive_patch_count": lambda m: int(np.count_nonzero(m.amyloid_val > 0.0)),
                "average_amyloid_value": lambda m: (
                    float(m.amyloid_val[m.amyloid_val > 0.0].mean())
                    if np.any(m.amyloid_val > 0.0)
                    else 0.0
                ),
                # Concentration mass and patch counts by amyloid size class.
                "amyloid_monomers_population": lambda m: float(np.sum((m.amyloid_val[(m.amyloid_val < 2) & (m.amyloid_val > 0.0)]))),
                "amyloid_polymers_population": lambda m: float(np.sum((m.amyloid_val[(m.amyloid_val >= 2) & (m.amyloid_val < 4.0)]))),
                "amyloid_plaques_population": lambda m: float(np.sum(m.amyloid_val[m.amyloid_val >= 4.0])),
                "amyloid_monomer_patch_count": lambda m: int(np.count_nonzero((m.amyloid_val < 2.0) & (m.amyloid_val > 0.0))),
                "amyloid_polymer_patch_count": lambda m: int(np.count_nonzero((m.amyloid_val >= 2.0) & (m.amyloid_val < 4.0))),
                "amyloid_plaque_patch_count": lambda m: int(np.count_nonzero(m.amyloid_val >= 4.0)),
                # --- Genotype metrics ---
                "microglia_TREM2_mutant": lambda m: sum(
                    1 for mg in m.microglia
                    if mg.pos is not None and getattr(mg, "trem2_is_mutant", False)
                ),

                # --- Toxic lipid metric ---
                "total_neuron_toxic_lipids": lambda m: float(sum(
                    getattr(n, "toxic_lipids", 0.0)
                    for n in m.neurons if n.pos is not None
                )),
                "neurons_with_toxic_lipids": lambda m: sum(
                    1 for n in m.neurons
                    if n.pos is not None and n.toxic_lipids > 0.01
                ),
                "LDAM_fraction": lambda m: (
                    sum(
                        1 for mg in m.microglia
                        if mg.pos is not None and getattr(mg, "isLDAM", False)
                    )
                    / max(1, sum(1 for mg in m.microglia if mg.pos is not None))
                ),

                "DAM_fraction": lambda m: (
                    sum(
                        1 for mg in m.microglia
                        if mg.pos is not None and getattr(mg, "isDAM", False)
                    )
                    / max(1, sum(1 for mg in m.microglia if mg.pos is not None))
                ),

                "TREM2_mutant_fraction": lambda m: (
                    sum(
                        1 for mg in m.microglia
                        if mg.pos is not None and getattr(mg, "trem2_is_mutant", False)
                    )
                    / max(1, sum(1 for mg in m.microglia if mg.pos is not None))
                ),
            }
        )

    # -------- helper functions --------

    def next_id(self) -> int:
        """ Return the next available unique ID for an agent. """
        self._uid += 1
        return self._uid

    def _place_random(self, agent):
        """ Place an agent at a random empty location on the grid."""
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        self.grid.place_agent(agent, (x, y))

    def _torus_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """ Provide torus-aware distance between two points a and b on the grid. """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.grid.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return (dx * dx + dy * dy) ** 0.5

    def remove_neuron(self, neuron: Neuron):
        """ Remove a neuron from the grid and set its position to None. """
        if neuron.pos is not None:
            self.grid.remove_agent(neuron)
            neuron.pos = None

    def _diffuse_inflammation(self):
        """
        Diffuse neuron-sourced DAMAGE (chemotaxis cue) outward
        from residence centers.
        """
        coords = list(zip(*np.where(self.residence)))
        for x, y in coords:
            r = self.res_curr[x, y]
            if r <= self.damage_radius - 1:
                xs = range(x - r, x + r + 1)
                ys = range(y - r, y + r + 1)
                for xi in xs:
                    for yi in ys:
                        xi2 = xi % self.width
                        yi2 = yi % self.height
                        if max(abs(xi - x), abs(yi - y)) <= r:
                            self.damage_val[xi2, yi2] += 1.0
                self.res_curr[x, y] = r + 1

    def _dismantle_inflammation(self):
        """
        Shrink DAMAGE around patches flagged as 'dismantling'
        (triggered by M2 repair).
        """
        coords = list(zip(*np.where(self.dismantling)))
        to_stop = []
        for x, y in coords:
            r = self.dis_curr[x, y]
            xs = range(x - r, x + r + 1)
            ys = range(y - r, y + r + 1)
            for xi in xs:
                for yi in ys:
                    xi2 = xi % self.width
                    yi2 = yi % self.height
                    if max(abs(xi - x), abs(yi - y)) <= r:
                        self.damage_val[xi2, yi2] = max(
                            0.0, self.damage_val[xi2, yi2] - 1.0
                        )
            r -= 1
            if r <= 0:
                to_stop.append((x, y))
            else:
                self.dis_curr[x, y] = r
        for x, y in to_stop:
            self.dismantling[x, y] = False
            self.dis_curr[x, y] = 0

    def _decay_signals(self):
        """
        Per-tick decay of both cytokine fields.
        This is what makes pro/anti signals die out over time when not replenished.
        """
        if self.pro_decay > 0.0:
            self.pro_inflam_val *= (1.0 - self.pro_decay)
        if self.anti_decay > 0.0:
            self.anti_inflam_val *= (1.0 - self.anti_decay)

        # clamp numerical negatives
        np.maximum(self.pro_inflam_val, 0.0, out=self.pro_inflam_val)
        np.maximum(self.anti_inflam_val, 0.0, out=self.anti_inflam_val)
        
    def diffuse_move_amyloid(self):
        """
        Per-tick diffusion of amyloid patches, with some random movement to represent plaque dynamics. 
        If levels are too low, they will dissipate; if they are high, they can grow and spread.
        Amyloid can also have a baseline production rate and a clearance mechanism (not implemented here but could be added).
        """
        diffusion_rate = self.diffusion_rate  # how much amyloid diffuses to neighboring cells per tick
        growth_rate = self.amyloid_growth  # logistic amyloid growth rate per tick
        carrying_capacity = self.amyloid_carrying_capacity
        decay_rate = self.amyloid_decay  # how much amyloid decays per tick

        new_amyloid_val = np.copy(self.amyloid_val)

        for x in range(self.width):
            for y in range(self.height):
                val = self.amyloid_val[x, y]
                if val > 0:
                    # Diffusion to neighbors
                    neighbors = self.grid.get_neighborhood((x, y), moore=True, include_center=False)
                    for nx, ny in neighbors:
                        transfer = val * diffusion_rate / len(neighbors)
                        new_amyloid_val[nx, ny] += transfer
                        new_amyloid_val[x, y] -= transfer
                    
                    # Movement: with some probability, the amyloid patch can move randomly to a neighboring cell, representing plaque dynamics. The amount that moves can be a fraction of the current value.
                    if self.random.random() < 0.1:
                        dx = self.random.choice([-1, 0, 1])
                        dy = self.random.choice([-1, 0, 1])
                        new_x = (x + dx) % self.width
                        new_y = (y + dy) % self.height
                        move_amount = val * 0.05  # amount to move
                        new_amyloid_val[new_x, new_y] += move_amount
                        new_amyloid_val[x, y] -= move_amount
                    
                    # Logistic growth: growth slows as local amyloid approaches K.
                    if carrying_capacity > 0.0:
                        logistic_factor = 1.0 - (val / carrying_capacity)
                        new_amyloid_val[x, y] += val * growth_rate * logistic_factor
                    
                    # Decay
                    new_amyloid_val[x, y] -= val * decay_rate

        # Clamp values to non-negative
        np.maximum(new_amyloid_val, 0.0, out=new_amyloid_val)
        if carrying_capacity > 0.0:
            np.minimum(new_amyloid_val, carrying_capacity, out=new_amyloid_val)
        self.amyloid_val = new_amyloid_val

    def step(self):
        """ Step logic for all agents: 
            - Microglia act (move, interact, emit cytokines)
            - Astrocytes reshape fields, act on neurons, optionally recruit microglia
            - Neurons update based on local pro/anti fields
            - Spatial expansion/retraction of damage every 5 ticks
            - Decay of cytokine fields
        """
        # 1) microglia act (move, interact, emit cytokines)
        for mg in self.microglia:
            mg.step()

        # 2) astrocytes reshape fields, act on neurons, optionally recruit microglia
        for a in self.astrocytes:
            a.step()

        # 3) neurons update based on local pro/anti fields
        for n in self.neurons:
            n.step()

        # 4) spatial expansion/retraction of damage every 5 ticks
        if self.steps % 5 == 0:
            self._diffuse_inflammation()
            self._dismantle_inflammation()

        # 5) decay of cytokine fields
        self._decay_signals()
        
        # 6) diffuse and move amyloid patches
        self.diffuse_move_amyloid()

        self.datacollector.collect(self)

    def all_damaged_cleared(self) -> bool:
        """ Helper to check if all damaged neurons have been cleared (used for stopping condition in some experiments). """
        return not any((n.pos is not None) and n.damaged for n in self.neurons)
    
    def all_plaques_cleared(self) -> bool:
        """ Helper to check if all amyloid plaques have been cleared (used for stopping condition in some experiments). """
        return np.sum(self.amyloid_val >= 4.0) == 0
  
  
def run_sim(
    steps=500,
    width=33,
    height=33,
    m0=3,
    m1=8,
    m2=2,
    h_neurons=10,
    d_neurons=10,
    eat=0.70,
    base_eat=None,
    sense=0.50,
    base_sense=None,
    damage=0.005,
    ndist=5,
    radius=3,
    seed=None,

    # astrocyte knobs
    a0_astro=10,
    a1_astro=5,
    a2_astro=5,
    astro_radius=2,
    recruit_thresh=3.0,
    recruit_prob=0.01,

    # lipid knobs
    neuron_ld_production_rate=0.2,
    neuron_ld_packet_size=0.5,
    neuron_to_astro_ld_transfer_prob=0.3,
    astro_ld_oxidation_rate=0.3,
    microglia_ld_pro_inflam_threshold=5.0,

    # ---------------- Neuron transition knobs ----------------
    damage_ratio_thresh=0.6,
    healthy_ratio_thresh=0.2,
    healthy_chance=0.20,
    damage_to_death_ticks=20,
    death_ratio_thresh=0.9,
    death_chance=0.25,

    # ---------------- Microglia phenotype signal knobs ----------------
    homeo_signal_thresh=1.0,
    pro_inflam_signal_thresh=1.05,
    anti_inflam_signal_thresh=0.95,
    homeo_chance=0.01,
    pro_inflam_chance=0.2,
    anti_inflam_chance=0.2,
    m0_resolution_ticks=10,

    # ---------------- Astrocyte phenotype signal knobs ----------------
    astro_homeo_signal_thresh=0.5,
    astro_pro_inflam_signal_thresh=1.05,
    astro_anti_inflam_signal_thresh=0.95,
    astro_homeo_chance=0.01,
    astro_pro_inflam_chance=0.3,
    astro_anti_inflam_chance=0.3,
    a0_resolution_ticks=10,

    # ---------------- Field decay knobs ----------------
    pro_decay=0.20,
    anti_decay=0.10,

    # ---------------- lipid dynamics knobs ----------------
    neuron_ld_damage_boost=0.5,
    neuron_ld_death_boost=1.0,
    neuron_to_astro_ld_transfer_radius=2,
    astro_ld_to_anti_inflam=0.05,
    microglia_ld_from_phagocytosis=0.5,
    
    # ldam
    microglia_ldam_inflam_boost=0.5,
    ldam_neuron_toxic_lipid_transfer=0.1,
    ldam_threshold=5.0,
    
    # dam
    dam_activation_threshold=3.0,
    dam_activation_ticks=5,  
    
    # ---------------- Amyloid dynamics knobs ----------------
    initial_amyloid_patches=5,
    amyloid_decay=0.05,
    amyloid_growth=0.1,
    amyloid_carrying_capacity=5.0,
    amyloid_diffusion=0.1,
    amyloid_fibril_thresh=2.0,
    microglia_ld_from_amyloid=0.5,
    microglia_amyloid_clearance=0.1,
    microglia_lipid_impairment_factor=0.05,
    microglia_amyloid_impairment_factor=0.5,
    
    apoe_genotype="3/3",
    trem2_mutation_rate=0.0,
    trem2_mutant_activity=0.5,
):
    """ Run a Microglia-Astrocyte-Neuron visual simulation located on a slice of the hippocampus with the provided parameters and return the model and collected data. 

    Args:
        steps (int, optional): Maximum number of steps to run the simulation. Defaults to 500.
        width (int, optional): Width of the grid. Defaults to 33.
        height (int, optional): Height of the grid. Defaults to 33.
        m0 (int, optional): Initial number of M0 microglia. Defaults to 3.
        m1 (int, optional): Initial number of M1 microglia. Defaults to 8.
        m2 (int, optional): Initial number of M2 microglia. Defaults to 2.
        h_neurons (int, optional): Initial number of healthy neurons. Defaults to 10.
        d_neurons (int, optional): Initial number of damaged neurons. Defaults to 10.
        eat (float, optional): Base probability of microglia successfully eating a damaged neuron. Defaults to 0.70.
        base_eat (float, optional): Optional baseline used when recalculating effective phagocytosis after impairment. Defaults to eat.
        sense (float, optional): Base efficiency of microglia sensing nearby damage. Defaults to 0.50.
        base_sense (float, optional): Optional baseline used when recalculating effective sensing after impairment. Defaults to sense.
        damage (float, optional): Base chance of a healthy neuron becoming damaged each tick. Defaults to 0 .005.
        ndist (int, optional): Distance at which microglia can sense and eat damaged neurons. Defaults to 5.
        radius (int, optional): Radius of damage diffusion from damaged neurons. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to None.    
        a0_astro (int, optional): Initial number of A0 astrocytes. Defaults to 10.
        a1_astro (int, optional): Initial number of A1 astrocytes. Defaults to 5.
        a2_astro (int, optional): Initial number of A2 astrocytes. Defaults to 5.
        astro_radius (int, optional): Radius of astrocyte influence on the grid. Defaults to 2.
        a0_resolution_ticks (int, optional): Number of low-signal ticks it takes for an A1 or A2 astrocyte to resolve back to A0. Defaults to 10.
        recruit_thresh (float, optional): Pro-inflammatory signal threshold for astrocytes to recruit microglia. Defaults to 3.0.
        recruit_prob (float, optional): Probability of recruiting a microglia when the pro-inflammatory signal threshold is met. Defaults to 0.01.
        neuron_ld_production_rate (float, optional): Rate at which neurons produce lipid droplets when damaged. Defaults to 0.2.
        neuron_ld_packet_size (float, optional): Amount of lipid droplets produced in each packet when a neuron produces LDs. Defaults to 0.5.
        neuron_to_astro_ld_transfer_prob (float, optional): Probability of damaged neurons transferring lipid droplets to nearby astrocytes. Defaults to 0.3.
        astro_ld_oxidation_rate (float, optional): Rate at which astrocytes can oxidize (burn) lipid droplets, reducing their pro-inflammatory effect. Defaults to 0.3.
        microglia_ld_pro_inflam_threshold (float, optional): Threshold of lipid droplets in microglia that boosts their pro-inflammatory signaling. Defaults to 5.0.
        damage_ratio_thresh (float, optional): Ratio of damage signal to healthy signal that increases the chance of a neuron transitioning from healthy to damaged. Defaults to 0.6.
        healthy_ratio_thresh (float, optional): Ratio of healthy signal to damage signal that increases the chance of a neuron transitioning from damaged back to healthy. Defaults to 0.2.
        healthy_chance (float, optional): Base chance of a damaged neuron transitioning back to healthy when the healthy ratio threshold is met. Defaults to 0.20.
        damage_to_death_ticks (int, optional): Number of ticks a damaged neuron must remain damaged before it can transition to dead. Defaults to 20.
        death_ratio_thresh (float, optional): Ratio of damage signal to healthy signal that increases the chance of a neuron transitioning from damaged to dead. Defaults to 0.9.
        death_chance (float, optional): Base chance of a damaged neuron transitioning to dead when the death ratio threshold is met and it has been damaged for at least damage_to_death_ticks. Defaults to 0.25.
        homeo_signal_thresh (float, optional): Threshold of pro/anti-inflammatory signal balance that increases the chance of a microglia transitioning to the homeostatic M0 phenotype. Defaults to 1.0.
        pro_inflam_signal_thresh (float, optional): Pro/anti ratio threshold above which microglia are biased toward M1. Defaults to 1.05.
        anti_inflam_signal_thresh (float, optional): Pro/anti ratio threshold below which microglia are biased toward M2. Defaults to 0.95.
        homeo_chance (float, optional): Base chance of a microglia transitioning to the M0 phenotype when the homeostatic signal threshold is met. Defaults to 0.01.
        pro_inflam_chance (float, optional): Base chance of a microglia transitioning to the M1 phenotype when the pro-inflammatory signal threshold is met. Defaults to 0.2.
        anti_inflam_chance (float, optional): Base chance of a microglia transitioning to the M2 phenotype when the anti-inflammatory signal threshold is met. Defaults to 0.2.
        m0_resolution_ticks (int, optional): Number of low-signal ticks an M1 or M2 microglia must remain in a resolving environment before it can transition back to M0. Defaults to 10.
        astro_homeo_signal_thresh (float, optional): Threshold of pro/anti-inflammatory signal balance that increases the chance of an astrocyte transitioning to the homeostatic A0 phenotype. Defaults to 0.5.
        astro_pro_inflam_signal_thresh (float, optional): Pro/anti ratio threshold above which astrocytes are biased toward A1. Defaults to 1.05.
        astro_anti_inflam_signal_thresh (float, optional): Pro/anti ratio threshold below which astrocytes are biased toward A2. Defaults to 0.95.
        astro_homeo_chance (float, optional): Base chance of an astrocyte transitioning to the A0 phenotype when the homeostatic signal threshold is met. Defaults to 0.01.
        astro_pro_inflam_chance (float, optional): Base chance of an astrocyte transitioning to the A1 phenotype when the pro-inflammatory signal threshold is met. Defaults to 0.3.
        astro_anti_inflam_chance (float, optional): Base chance of an astrocyte transitioning to the A2 phenotype when the anti-inflammatory signal threshold is met. Defaults to 0.3.
        pro_decay (float, optional): Per-tick decay rate of the pro-inflammatory signal. Defaults to 0.20.
        anti_decay (float, optional): Per-tick decay rate of the anti-inflammatory signal. Defaults to 0.10.
        neuron_ld_damage_boost (float, optional): Amount by which each unit of lipid droplets in a neuron boosts the damage signal it emits. Defaults to 0.5.
        neuron_ld_death_boost (float, optional): Amount by which each unit of lipid droplets in a neuron boosts the chance of it transitioning from damaged to dead. Defaults to 1.0.
        neuron_to_astro_ld_transfer_radius (int, optional): Radius within which damaged neurons can transfer lipid droplets to nearby astrocytes. Defaults to 2.
        astro_ld_to_anti_inflam (float, optional): Amount by which each unit of lipid droplets in an astrocyte boosts the anti-inflammatory signal it emits. Defaults to 0.05.
        microglia_ld_from_phagocytosis (float, optional): Amount of lipid droplets a microglia gains from successfully eating a damaged neuron. Defaults to 0.5.
        microglia_ldam_inflam_boost (float, optional): Amount by which each unit of lipid droplets in a microglia boosts its pro    -inflammatory signaling when it becomes an LDAM. Defaults to 0.5.
        ldam_neuron_toxic_lipid_transfer (float, optional): Amount of toxic lipids transferred from a microglia that has become an LDAM to nearby neurons, boosting their damage signal and chance of transitioning to damaged. Defaults to 0.1.
        ldam_threshold (float, optional): Threshold of lipid droplets in a microglia at which it transitions to the dysfunctional LDAM state. Defaults to 5.0.
        dam_activation_threshold (float, optional): Threshold of pro-inflammatory signal at which a microglia transitions to the DAM state. Defaults to 3.0.
        dam_activation_ticks (int, optional): Number of ticks a microglia must remain above the DAM activation threshold before it transitions to the DAM state. Defaults to 5
        initial_amyloid_patches (int, optional): Number of initial amyloid patches to seed in the grid. Defaults to 5.
        amyloid_decay (float, optional): Per-tick decay rate of amyloid patches. Defaults to 0.05.
        amyloid_growth (float, optional): Per-tick logistic growth rate of amyloid patches. Defaults to 0.1.
        amyloid_carrying_capacity (float, optional): Per-patch carrying capacity K used to cap logistic amyloid growth. Defaults to 5.0.
        amyloid_diffusion (float, optional): Per-tick diffusion rate of amyloid patches to neighboring cells. Defaults to 0.1.
        amyloid_fibril_thresh (float, optional): Amyloid concentration threshold above which the amyloid is considered fibrillar and more toxic. Defaults to 2.0.   
        microglia_ld_from_amyloid (float, optional): Amount of lipid droplets a microglia gains from being exposed to amyloid patches, boosting its pro-inflammatory signaling. Defaults to 0.5.
        microglia_amyloid_clearance (float, optional): Amount by which a microglia can reduce the amyloid concentration in a patch when it interacts with it, representing clearance. Defaults to 0.1.
        microglia_lipid_impairment_factor (float, optional): Baseline strength of lipid-burden impairment on microglial sensing and phagocytosis. Defaults to 0.05.
        microglia_amyloid_impairment_factor (float, optional): Extra multiplier on lipid-burden impairment while amyloid exposure persists. Defaults to 0.5.
        apoe_genotype (str, optional): APOE genotype of the system, which can influence various parameters such as amyloid dynamics, microglia behavior, and neuron vulnerability. Defaults to "3/3".
        trem2_mutation_rate (float, optional): Proportion of microglia that have a TREM2 mutation, which impairs their function. Defaults to 0.0.
        trem2_mutant_activity (float, optional): Relative activity level of TREM2 mutant microglia compared to wild-type, affecting their sensing and eating capabilities. Defaults to 0.5.
    """
    model = MicrogliaNeuronModel(
        width=width,
        height=height,
        init_m0_microglia=m0,
        init_m1_microglia=m1,
        init_m2_microglia=m2,
        init_healthy_neuron=h_neurons,
        init_damaged_neuron=d_neurons,
        init_dead_neuron=0,
        eat_probability=eat,
        base_eat_probability=base_eat,
        sensing_efficiency=sense,
        base_sensing_efficiency=base_sense,
        damage_chance=damage,
        neuron_distance=ndist,
        damage_radius=radius,
        seed=seed,

        # astrocytes
        init_a0_astrocytes=a0_astro,
        init_a1_astrocytes=a1_astro,
        init_a2_astrocytes=a2_astro,
        astro_coverage_radius=astro_radius,
        a0_resolution_ticks=a0_resolution_ticks,

        recruitment_threshold=recruit_thresh,
        microglia_recruitment_prob=recruit_prob,

        # lipid knobs you already had
        neuron_ld_production_rate=neuron_ld_production_rate,
        neuron_ld_packet_size=neuron_ld_packet_size,
        neuron_to_astro_ld_transfer_prob=neuron_to_astro_ld_transfer_prob,
        astro_ld_oxidation_rate=astro_ld_oxidation_rate,
        microglia_ld_pro_inflam_threshold=microglia_ld_pro_inflam_threshold,

        # neuron transition
        damage_ratio_thresh=damage_ratio_thresh,
        healthy_ratio_thresh=healthy_ratio_thresh,
        healthy_chance=healthy_chance,
        damage_to_death_ticks=damage_to_death_ticks,
        death_ratio_thresh=death_ratio_thresh,
        death_chance=death_chance,

        # microglia phenotype signals + resolution
        homeo_signal_thresh=homeo_signal_thresh,
        pro_inflam_signal_thresh=pro_inflam_signal_thresh,
        anti_inflam_signal_thresh=anti_inflam_signal_thresh,
        homeo_chance=homeo_chance,
        pro_inflam_chance=pro_inflam_chance,
        anti_inflam_chance=anti_inflam_chance,
        m0_resolution_ticks=m0_resolution_ticks,

        # Astrocyte phenotype signals
        astro_homeo_signal_thresh=astro_homeo_signal_thresh,
        astro_pro_inflam_signal_thresh=astro_pro_inflam_signal_thresh,
        astro_anti_inflam_signal_thresh=astro_anti_inflam_signal_thresh,
        astro_homeo_chance=astro_homeo_chance,
        astro_pro_inflam_chance=astro_pro_inflam_chance,
        astro_anti_inflam_chance=astro_anti_inflam_chance,

        # field decay
        pro_decay=pro_decay,
        anti_decay=anti_decay,

        # extra lipid dynamics
        neuron_ld_damage_boost=neuron_ld_damage_boost,
        neuron_ld_death_boost=neuron_ld_death_boost,
        neuron_to_astro_ld_transfer_radius=neuron_to_astro_ld_transfer_radius,
        astro_ld_to_anti_inflam=astro_ld_to_anti_inflam,
        microglia_ld_from_phagocytosis=microglia_ld_from_phagocytosis,
        
        # ldam
        microglia_ldam_inflam_boost=microglia_ldam_inflam_boost,
        ldam_neuron_toxic_lipid_transfer=ldam_neuron_toxic_lipid_transfer,
        ldam_threshold=ldam_threshold,
        
        # dam
        dam_activation_threshold=dam_activation_threshold,
        dam_activation_ticks=dam_activation_ticks,      

        # amyloid dynamics
        initial_amyloid_patches=initial_amyloid_patches,
        amyloid_decay=amyloid_decay,
        amyloid_growth=amyloid_growth,
        amyloid_carrying_capacity=amyloid_carrying_capacity,
        amyloid_diffusion=amyloid_diffusion,
        amyloid_fibril_thresh=amyloid_fibril_thresh,
        microglia_ld_from_amyloid=microglia_ld_from_amyloid,
        microglia_amyloid_clearance=microglia_amyloid_clearance,
        microglia_lipid_impairment_factor=microglia_lipid_impairment_factor,
        microglia_amyloid_impairment_factor=microglia_amyloid_impairment_factor,
        
        apoe_genotype=apoe_genotype,
        trem2_mutation_rate=trem2_mutation_rate,
        trem2_mutant_activity=trem2_mutant_activity,
    )

    for _ in range(steps):
        #if not any((n.pos is not None) and n.damaged for n in model.neurons):
        #    break
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return model, df

def main():
    """ A simple command-line interface to run the Microglia-Neuron ABM with custom parameters and visualize results with matplotlib. """
    ap = argparse.ArgumentParser(description="Microglia–Neuron ABM")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=int, default=33)
    ap.add_argument("--height", type=int, default=33)
    ap.add_argument("--m0", type=int, default=5)
    ap.add_argument("--m1", type=int, default=5)
    ap.add_argument("--m2", type=int, default=5)
    ap.add_argument("--h_neurons", type=int, default=10)
    ap.add_argument("--d_neurons", type=int, default=10)
    ap.add_argument("--eat", type=float, default=0.70)
    ap.add_argument("--base_eat", type=float, default=None)
    ap.add_argument("--sense", type=float, default=0.50)
    ap.add_argument("--base_sense", type=float, default=None)
    ap.add_argument("--damage", type=float, default=0.005)
    ap.add_argument("--ndist", type=int, default=5)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    # astrocyte flags (CLI)
    ap.add_argument("--a0_astro", type=int, default=10)
    ap.add_argument("--a1_astro", type=int, default=5)
    ap.add_argument("--a2_astro", type=int, default=5)
    ap.add_argument("--astro_radius", type=int, default=2)
    ap.add_argument("--a0_resolution_ticks", type=int, default=10)
    ap.add_argument("--recruit_thresh", type=float, default=3.0)
    ap.add_argument("--recruit_prob", type=float, default=0.01)
    # lipid knobs
    ap.add_argument("--neuron_ld_production_rate", type=float, default=0.2)
    ap.add_argument("--neuron_ld_packet_size", type=float, default=0.5)
    ap.add_argument("--neuron_to_astro_ld_transfer_prob", type=float, default=0.3)
    ap.add_argument("--astro_ld_oxidation_rate", type=float, default=0.3)
    ap.add_argument("--microglia_ld_pro_inflam_threshold", type=float, default=5.0)
    # neuron transition knobs
    ap.add_argument("--damage_ratio_thresh", type=float, default=0.6)
    ap.add_argument("--healthy_ratio_thresh", type=float, default=0.2)
    ap.add_argument("--healthy_chance", type=float, default=0.20)
    ap.add_argument("--damage_to_death_ticks", type=int, default=20)
    ap.add_argument("--death_ratio_thresh", type=float, default=0.9)
    ap.add_argument("--death_chance", type=float, default=0.25)
    # microglia phenotype signal knobs
    ap.add_argument("--homeo_signal_thresh", type=float, default=1.0)
    ap.add_argument("--pro_inflam_signal_thresh", type=float, default=1.05)
    ap.add_argument("--anti_inflam_signal_thresh", type=float, default=0.95)
    ap.add_argument("--homeo_chance", type=float, default=0.01)
    ap.add_argument("--pro_inflam_chance", type=float, default=0.2)
    ap.add_argument("--anti_inflam_chance", type=float, default=0.2)
    ap.add_argument("--m0_resolution_ticks", type=int, default=10)
    # astrocyte phenotype signal knobs
    ap.add_argument("--astro_homeo_signal_thresh", type=float, default=0.5)
    ap.add_argument("--astro_pro_inflam_signal_thresh", type=float, default=1.05)
    ap.add_argument("--astro_anti_inflam_signal_thresh", type=float, default=0.95)
    ap.add_argument("--astro_homeo_chance", type=float, default=0.01)
    ap.add_argument("--astro_pro_inflam_chance", type=float, default=0.3)
    ap.add_argument("--astro_anti_inflam_chance", type=float, default=0.3)
    # field decay knobs
    ap.add_argument("--pro_decay", type=float, default=0.20)
    ap.add_argument("--anti_decay", type=float, default=0.10)
    # lipid dynamics knobs
    ap.add_argument("--neuron_ld_damage_boost", type=float, default=0.5)
    ap.add_argument("--neuron_ld_death_boost", type=float, default=1.0)
    ap.add_argument("--neuron_to_astro_ld_transfer_radius", type=int, default=2)
    ap.add_argument("--astro_ld_to_anti_inflam", type=float, default=0.05)
    ap.add_argument("--microglia_ld_from_phagocytosis", type=float, default=0.5)
    # ldam
    ap.add_argument("--microglia_ldam_inflam_boost", type=float, default=0.5)
    ap.add_argument("--ldam_neuron_toxic_lipid_transfer", type=float, default=0.1)
    ap.add_argument("--ldam_threshold", type=float, default=5.0)
    # dam
    ap.add_argument("--dam_activation_threshold", type=float, default=3.0)
    ap.add_argument("--dam_activation_ticks", type=int, default=5)
    # amyloid dynamics knobs
    ap.add_argument("--initial_amyloid_patches", type=int, default=5)
    ap.add_argument("--amyloid_decay", type=float, default=0.05)
    ap.add_argument("--amyloid_growth", type=float, default=0.1)
    ap.add_argument("--amyloid_carrying_capacity", type=float, default=5.0)
    ap.add_argument("--amyloid_diffusion", type=float, default=0.1)
    ap.add_argument("--amyloid_fibril_thresh", type=float, default=2.0)
    ap.add_argument("--microglia_ld_from_amyloid", type=float, default=0.5)
    ap.add_argument("--microglia_amyloid_clearance", type=float, default=0.1)
    ap.add_argument("--microglia_lipid_impairment_factor", type=float, default=0.05)
    ap.add_argument("--microglia_amyloid_impairment_factor", type=float, default=0.5)
    ap.add_argument("--apoe_genotype", type=str, default="3/3")
    ap.add_argument("--trem2_mutation_rate", type=float, default=0.0)
    ap.add_argument("--trem2_mutant_activity", type=float, default=0.5)
    ap.add_argument("--output_csv", type=str, default="results.csv")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--summary_png", type=str, default=None)
    ap.add_argument("--no_plot", action="store_true")

    args, _ = ap.parse_known_args()
    run_kwargs = vars(args).copy()
    output_csv = run_kwargs.pop("output_csv")
    no_csv = run_kwargs.pop("no_csv")
    summary_png = run_kwargs.pop("summary_png")
    no_plot = run_kwargs.pop("no_plot")

    model, df_raw = run_sim(**run_kwargs)

    # Normalize columns similar to the UI
    df = df_raw.copy()
    if "total_inflammation" not in df.columns:
        if "total_pro_inflammation" in df.columns and "total_anti_inflammation" in df.columns:
            df["total_inflammation"] = (
                df["total_pro_inflammation"] + df["total_anti_inflammation"]
            )
    if "mean_inflammation" not in df.columns:
        if "mean_pro_inflammation" in df.columns and "mean_anti_inflammation" in df.columns:
            df["mean_inflammation"] = (
                df["mean_pro_inflammation"] + df["mean_anti_inflammation"]
            )

    print("\nFinal snapshot:")
    print(df.tail(1).to_string(index=False))
    print(f"Stopped at tick {int(df['step'].iloc[-1])}")

    # Save the full dataframe as CSV when requested.
    if output_csv and not no_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

    if no_plot:
        return

    # NOTE: now 4 subplots to include astrocytes
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))

    # Neurons
    axs[0].plot(df["step"], df["healthy_neurons"], label="healthy")
    axs[0].plot(df["step"], df["damaged_neurons"], label="damaged")
    if "dead_neurons" in df.columns:
        axs[0].plot(df["step"], df["dead_neurons"], label="dead")
    if "neurons_total" in df.columns:
        axs[0].plot(df["step"], df["neurons_total"], label="total")
    axs[0].set_title("Neurons")
    axs[0].legend()

    # Inflammation (pro / anti / total)
    if "total_pro_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_pro_inflammation"], label="pro")
    if "total_anti_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_anti_inflammation"], label="anti")
    if "total_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_inflammation"], label="total")
    axs[1].set_title("Inflammation")
    axs[1].legend()

    # Microglia
    if "microglia_total" in df.columns:
        axs[2].plot(df["step"], df["microglia_total"], label="total")
    if "microglia_M0" in df.columns:
        axs[2].plot(df["step"], df["microglia_M0"], label="M0")
    if "microglia_M1" in df.columns:
        axs[2].plot(df["step"], df["microglia_M1"], label="M1")
    if "microglia_M2" in df.columns:
        axs[2].plot(df["step"], df["microglia_M2"], label="M2")
    axs[2].set_title("Microglia")
    axs[2].legend()

    # Astrocytes
    if "astrocytes_total" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_total"], label="total")
    if "astrocytes_A0" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A0"], label="A0")
    if "astrocytes_A1" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A1"], label="A1")
    if "astrocytes_A2" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A2"], label="A2")
    axs[3].set_title("Astrocytes")
    axs[3].legend()

    plt.tight_layout()
    if summary_png or os.environ.get("SOLARA_SERVER"):
        out_path = summary_png or "summary.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved summary plot to {out_path}")
    else:
        plt.show()
        plt.close(fig)
        
# If this script is run directly, execute the main function
if __name__ == "__main__":
    main() # takes CLI args for custom runs, or just runs with defaults if no args provided
