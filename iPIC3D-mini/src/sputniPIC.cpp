/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    double iMoverGPU, eMoverGPU = 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // Save particle states before movers
        particles *part_cpu = new particles[param.ns];
        particles *part_gpu = new particles[param.ns];
        
        // Allocate and copy particle data for CPU test
        for (int is=0; is < param.ns; is++){
            part_cpu[is] = part[is];
            part_cpu[is].x = new FPpart[part[is].npmax];
            part_cpu[is].y = new FPpart[part[is].npmax];
            part_cpu[is].z = new FPpart[part[is].npmax];
            part_cpu[is].u = new FPpart[part[is].npmax];
            part_cpu[is].v = new FPpart[part[is].npmax];
            part_cpu[is].w = new FPpart[part[is].npmax];
            for (int i=0; i < part[is].nop; i++){
                part_cpu[is].x[i] = part[is].x[i];
                part_cpu[is].y[i] = part[is].y[i];
                part_cpu[is].z[i] = part[is].z[i];
                part_cpu[is].u[i] = part[is].u[i];
                part_cpu[is].v[i] = part[is].v[i];
                part_cpu[is].w[i] = part[is].w[i];
            }
        }
        
        // Allocate and copy particle data for GPU test
        for (int is=0; is < param.ns; is++){
            part_gpu[is] = part[is];
            part_gpu[is].x = new FPpart[part[is].npmax];
            part_gpu[is].y = new FPpart[part[is].npmax];
            part_gpu[is].z = new FPpart[part[is].npmax];
            part_gpu[is].u = new FPpart[part[is].npmax];
            part_gpu[is].v = new FPpart[part[is].npmax];
            part_gpu[is].w = new FPpart[part[is].npmax];
            for (int i=0; i < part[is].nop; i++){
                part_gpu[is].x[i] = part[is].x[i];
                part_gpu[is].y[i] = part[is].y[i];
                part_gpu[is].z[i] = part[is].z[i];
                part_gpu[is].u[i] = part[is].u[i];
                part_gpu[is].v[i] = part[is].v[i];
                part_gpu[is].w[i] = part[is].w[i];
            }
        }
        
        
        std::cout << "\n=== CPU Mover ===" << std::endl;
        iMover = cpuSecond(); 
        for (int is=0; is < param.ns; is++)
            mover_PC(&part_cpu[is],&field,&grd,&param);
        eMover += (cpuSecond() - iMover); 
        
        // GPU mover
        std::cout << "\n=== GPU Mover ===" << std::endl;
        iMoverGPU = cpuSecond(); 
        for (int is=0; is < param.ns; is++)
            mover_PC_gpu(&part_gpu[is],&field,&grd,&param);
        eMoverGPU += (cpuSecond() - iMoverGPU);
        
        std::cout << "\n=== Validation: CPU vs GPU ===" << std::endl;
        for (int is=0; is < param.ns; is++){
            double max_error_x = 0.0, max_error_y = 0.0, max_error_z = 0.0;
            double max_error_u = 0.0, max_error_v = 0.0, max_error_w = 0.0;
            
            for (int i=0; i < part[is].nop; i++){
                double err_x = fabs(part_cpu[is].x[i] - part_gpu[is].x[i]);
                double err_y = fabs(part_cpu[is].y[i] - part_gpu[is].y[i]);
                double err_z = fabs(part_cpu[is].z[i] - part_gpu[is].z[i]);
                double err_u = fabs(part_cpu[is].u[i] - part_gpu[is].u[i]);
                double err_v = fabs(part_cpu[is].v[i] - part_gpu[is].v[i]);
                double err_w = fabs(part_cpu[is].w[i] - part_gpu[is].w[i]);
                
                if (err_x > max_error_x) max_error_x = err_x;
                if (err_y > max_error_y) max_error_y = err_y;
                if (err_z > max_error_z) max_error_z = err_z;
                if (err_u > max_error_u) max_error_u = err_u;
                if (err_v > max_error_v) max_error_v = err_v;
                if (err_w > max_error_w) max_error_w = err_w;
            }
            
            double tolerance = 1e-5;
            bool passed = (max_error_x < tolerance && max_error_y < tolerance && max_error_z < tolerance &&
                          max_error_u < tolerance && max_error_v < tolerance && max_error_w < tolerance);
            
            if (passed) {
                std::cout << "GPU results match CPU within tolerance " << tolerance << std::endl;
            } else {
                std::cout << "GPU results differ from CPU beyond tolerance " << tolerance << std::endl;
            }
        }
        
        for (int is=0; is < param.ns; is++){
            for (int i=0; i < part[is].nop; i++){
                part[is].x[i] = part_gpu[is].x[i];
                part[is].y[i] = part_gpu[is].y[i];
                part[is].z[i] = part_gpu[is].z[i];
                part[is].u[i] = part_gpu[is].u[i];
                part[is].v[i] = part_gpu[is].v[i];
                part[is].w[i] = part_gpu[is].w[i];
            }
        }
        
        for (int is=0; is < param.ns; is++){
            delete[] part_cpu[is].x;
            delete[] part_cpu[is].y;
            delete[] part_cpu[is].z;
            delete[] part_cpu[is].u;
            delete[] part_cpu[is].v;
            delete[] part_cpu[is].w;
            delete[] part_gpu[is].x;
            delete[] part_gpu[is].y;
            delete[] part_gpu[is].z;
            delete[] part_gpu[is].u;
            delete[] part_gpu[is].v;
            delete[] part_gpu[is].w;
        }
        delete[] part_cpu;
        delete[] part_gpu;
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    } 
    
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    double iElaps = cpuSecond() - iStart;
    
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   CPU Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   GPU Mover Time / Cycle   (s) = " << eMoverGPU/param.ncycles << std::endl;
    std::cout << "   Speedup (CPU/GPU)            = " << eMover/eMoverGPU << "x" << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


