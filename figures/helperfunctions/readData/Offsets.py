import h5py
import numpy as np
import time
import os

class Offsets():

    def __init__(self, gpath, spath, sub_idx, fof_idx, num_gfiles, num_sfiles):
        
        if sub_idx == -1 and fof_idx == -1:
            return 

        self.grp_path = gpath
        self.snap_path = spath
        self.num_grp_files = num_gfiles
        self.num_snap_files = num_sfiles

        self.sub_idx = None
        self.fof_idx = None
        self._get_idx(sub_idx, fof_idx)

        self.gal_file = None
        self.gal_before = None
        self.part_files = None
        self.particles_in_gal = None
        self.particles_before_file = None
        self.particles_before_gal = None
        self.particles_per_file = None
        self.get_gal_file_num()

        return 

    #redefine sub_idx if it is given relative to the fof index
    #otherwise keep it the same as given
    def _get_idx(self, sub_idx, fof_idx):
        if sub_idx==-1 or fof_idx == -1:
            self.sub_idx = sub_idx
            self.fof_idx = fof_idx
        else:
            file_counter = 0
            for i in range(self.num_grp_files):
                if self.num_grp_files == 1:
                    file_name = self.grp_path
                else:
                    file_name = f"{self.grp_path}{i}.hdf5"
                with h5py.File(file_name, "r") as ofile:
                    halos_this_file = ofile['Header'].attrs['Ngroups_ThisFile']
                    file_counter += int(halos_this_file)

                    if file_counter > fof_idx:
                        fof_start_idx = int(ofile['Group/GroupFirstSub'][fof_idx - (file_counter - halos_this_file)])
                        break
            self.sub_idx = fof_start_idx + sub_idx
            self.fof_idx = -1
        return


    def check_if_offsets(self):
        if 'postprocessing' in os.listdir("/".join(self.snap_path.split("/")[:-3])):
            return True
        return False

    def get_saved_offsets(self): 

        return

    #get which file number our galaxy is on and the particle locations
    #probably not work if there are subhalos that are not in groups
    #may not work if there are fofs without subhalos before fofs with subhalos 
    def get_gal_file_num(self):
        dkey = "Group/GroupLenType"
        if self.sub_idx > -1:
            idx = self.sub_idx
            pkey = "Subhalo/SubhaloLenType"
            gkey = "Subhalo/SubhaloGrNr"
            is_subs = True
            hkey = "Nsubgroups_ThisFile"
            tkey = "Nsubgroups_Total"
        elif self.fof_idx > -1:
            idx = self.fof_idx
            pkey = "Group/GroupLenType"
            is_subs = False
            hkey = "Ngroups_ThisFile"
            tkey = "Ngroups_Total"
        else:
            return #not a single galaxy

        num_groups = 0
        num_gals=0 #number of subs/fofs in files before idx
        num_gals_groups = 0 #number of subhalos before our fof, using the fof groups to count
        gal_file=-1 #the file that the sub/fof is located on
        particles_before_gal = np.zeros(6) #number of particles of each type before the galaxy
        particles_in_gal = np.zeros(6) #number of particles on each type in the galaxy
        subs_still_to_go = None
        particles_before_file = np.zeros(6)
        for i in range(self.num_grp_files):
            if self.num_grp_files == 1:
                file_name = self.grp_path
            else:
                file_name = f"{self.grp_path}{i}.hdf5"
            with h5py.File(file_name, "r") as ofile:
                num_gals_this_file = int(ofile['Header'].attrs[hkey])
                num_groups_this_file = int(ofile['Header'].attrs['Ngroups_ThisFile'])

                if num_gals_this_file == 0: #if a file has no data in it - found in dm only unifrom box
                    continue

                #the number of subhalos that are already accounted for in this file
                #this will happen when the fof that a subhalo is in is not the first on the file 
                num_subs_this_file = 0

                #if the groups for the subs on this file have already been accounted for, then remove
                #the galaxies from the counters that are relative to the current file index 
                if subs_still_to_go is not None:
                    if subs_still_to_go >= num_gals_this_file:
                        subs_before_group -= num_gals_this_file
                        subs_still_to_go -= num_gals_this_file
                        num_gals += num_gals_this_file
                        continue

                if idx < num_gals+num_gals_this_file: #if the sub/fof is on this file
                    gal_file = i
                    
                    #get any particles on this file in fof groups before out sub's group
                    if is_subs:

                        #if we have had to get the number of subhalos before ours from a different file
                        if subs_still_to_go is not None:
                            particles_before_gal += np.sum(ofile['Subhalo/SubhaloLenType'][subs_before_group:subs_still_to_go], axis=0)
                            particles_in_gal = np.array(ofile[pkey][idx-num_gals])
                            break
                        #if the subhalo is on the same file as its fof
                        #account for any groups that are before our subhalo
                        else:
                            group_counter = 0
                            num_subs_this_file = ofile['Group/GroupFirstSub'][ofile[gkey][idx-num_gals]-num_groups]-num_gals

                            while num_groups < ofile['Subhalo/SubhaloGrNr'][idx-num_gals]:
                                particles_before_gal += np.array(ofile['Group/GroupLenType'][group_counter])
                                num_groups += 1
                                group_counter += 1


                    #get the number of particles in the current fof before our sub and the particles in
                    #the current fof/sub
                    particles_in_gal = np.array(ofile[pkey][idx-num_gals])
                    particles_before_gal += np.sum(ofile[pkey][num_subs_this_file:idx-num_gals], axis=0)
                    break

                #check if the group starts on this file but the subhalo is on a different one
                if is_subs:
                    nsubs = np.array(ofile['Group/GroupNsubs'])
                    if idx < num_gals+np.sum(nsubs) or (num_gals_groups >= num_gals and idx < num_gals_groups + np.sum(nsubs)):
                        #get the index into the subhalo's fof group
                        first_sub = np.array(ofile['Group/GroupFirstSub'])
                        group_li = np.arange(ofile['Header'].attrs['Ngroups_ThisFile'])
                        my_group_cut = (first_sub <= idx) & (first_sub != -1)
                        my_group = group_li[my_group_cut][-1]

                        #find the number of particles and subhalos before the target one
                        parts_before_my_group = np.sum(ofile['Group/GroupLenType'][:my_group], axis=0)
                        particles_before_gal += parts_before_my_group
                        subs_before_group = np.sum(nsubs[:my_group]) + num_gals_groups -num_gals - num_gals_this_file 
        
                        #figure out how many subs are on this file and subsequent files that we 
                        #still need to account for
                        subs_still_to_go = idx - num_gals - num_gals_this_file
                        subs_in_file = num_gals_this_file - (first_sub[my_group] - num_gals)

                        #subhalos before ours that are in the same fof but on a different file
                        if subs_in_file <=0: 
                            pass 
                        else:
                            last_parts = np.sum(ofile['Subhalo/SubhaloLenType'][-subs_in_file:], axis=0)
                            particles_before_gal += last_parts

                        num_gals += num_gals_this_file
                        continue
                    else:
                        num_gals_groups += np.sum(ofile['Group/GroupNsubs']) 
                
                #if the fof is not on this file add all particles on the file (parts in all fofs)
                particles_this_file = np.array(ofile[dkey])
                particles_before_gal += np.sum(particles_this_file, axis=0)
                num_gals += num_gals_this_file
                num_groups += num_groups_this_file

        self.gal_file = gal_file
        self.gal_before = num_gals

        self.particles_in_gal = particles_in_gal
        self.particles_before_file = particles_before_file
        self.particles_before_gal = particles_before_gal

        particle_files = np.zeros(6)
        num_parts_before = np.zeros(6)
        num_parts_before_file = np.zeros(6)
        particles_per_file = []
        done = [False]*6
        for i in range(self.num_grp_files):
            if False not in done:
                break
            if self.num_grp_files == 1:
                file_name = self.snap_path
            else:
                file_name = f"{self.snap_path}{i}.hdf5"
            with h5py.File(file_name, "r") as ofile:
                num_parts_file = np.array(ofile['Header'].attrs["NumPart_ThisFile"])
                particles_per_file.append(num_parts_file)
                for part in range(6):
                    if done[part]:
                        continue
                    if num_parts_file[part]+num_parts_before[part] >= particles_before_gal[part]+particles_in_gal[part]:
                        particle_files[part] = i
                        done[part] = True
                        num_parts_before_file[part] = particles_before_gal[part] - num_parts_before[part]
                    else:
                        num_parts_before[part] += num_parts_file[part]

        self.particles_per_file = np.array(particles_per_file)
        self.part_files = particle_files

        return

