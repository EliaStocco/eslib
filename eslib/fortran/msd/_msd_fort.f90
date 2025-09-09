!***********************************************************************
! Subroutine: shifted_msd
!
! Description:
! This subroutine calculates the mean squared displacement (MSD) 
! over M time shifts for a given set of atomic positions sampled over 
! nsnapshots. The displacement is computed relative to a reference 
! configuration and normalized by the number of valid snapshots used.
!
! Input Arguments:
! - positions(nsnapshots, natoms, 3): Real(8) array containing atomic 
!   positions at each snapshot. The dimensions represent:
!     nsnapshots - Number of snapshots (time steps).
!     natoms     - Number of atoms.
!     3          - Cartesian coordinates (x, y, z).
!
! - nsnapshots: Integer specifying the total number of snapshots.
! - natoms    : Integer specifying the number of atoms.
! - M         : Integer specifying the maximum time shift for MSD 
!               calculation.
! - verbose   : Logical flag; if .true., prints progress messages for
!               debugging and monitoring. Otherwise, runs silently.
!
! Output Argument:
! - delta_squared(M, natoms): Real(8) array where the element 
!   delta_squared(m, j) represents the average squared displacement 
!   of atom j after a time shift of m.
!
! Assumptions:
! - nsnapshots must be at least 2*M for meaningful results.
! - M and natoms must be greater than 0.
!
! Error Handling:
! - Stops execution if any input constraints are violated.
!
!***********************************************************************

subroutine shifted_msd(positions, delta_squared, verbose, nsnapshots, natoms, M)
    implicit none

    ! Input arguments
    integer, intent(in) :: nsnapshots, natoms, M
    logical, intent(in) :: verbose
    real(8), intent(in) :: positions(nsnapshots, natoms, 3)  ! (time_steps, natoms, 3)

    ! Output argument
    real(8), intent(inout) :: delta_squared(M, natoms)  ! (M, natoms)

    ! Local variables
    integer :: snapshot_idx, atom_idx, ref_idx
    real(8), dimension(natoms, 3) :: ref_positions  ! Reference configuration
    real(8) :: displacement_squared  ! Temporary variable for squared displacement
    integer :: valid_snapshots_count  ! Counter for valid snapshots processed
    integer :: progress  ! Progress variables

    !-------------------------------------------------------------------
    ! Input validation
    !-------------------------------------------------------------------
    if (nsnapshots < 2 * M) then
        print *, "Error: nsnapshots must be at least 2*M"
        stop
    end if
    if (M <= 0) then
        print *, "Error: M must be greater than 0"
        stop
    end if
    if (natoms <= 0) then
        print *, "Error: natoms must be greater than 0"
        stop
    end if

    ! if (verbose) print *, "Input validation successful"

    !-------------------------------------------------------------------
    ! Initialization
    !-------------------------------------------------------------------
    delta_squared(:,:) = 0.0  ! Reset the output array
    valid_snapshots_count = 0  ! Initialize counter for valid snapshots
    progress = 0              ! Initialize progress counter

    ! if (verbose) print *, "Initialization complete"

    !-------------------------------------------------------------------
    ! Main computation loop
    !-------------------------------------------------------------------
    do ref_idx = 1, M
        ! Reference configuration at snapshot ref_idx
        ref_positions = positions(ref_idx, :, :)  

        do snapshot_idx = ref_idx, ref_idx + M - 1
            if (snapshot_idx > nsnapshots) exit  ! Prevent out-of-bounds access

            do atom_idx = 1, natoms
                ! Compute squared displacement of atom atom_idx
                displacement_squared = sum((positions(snapshot_idx, atom_idx, :) - ref_positions(atom_idx, :))**2)
                delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
                    delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
            end do
        end do

        valid_snapshots_count = valid_snapshots_count + 1

        ! Update progress bar
        if (verbose) then
            progress = progress + 1
            if (progress == 1) then
                ! For the first iteration, print progress and move to a new line
                write(*, '(A)') ""  ! Newline after the first print
                write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"                
            else
                ! For subsequent iterations, update the progress on the same line
                write(*, '(A)', advance='no') char(13)  ! Return carriage to overwrite the previous progress
                write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"
            end if
        end if
    end do

    !-------------------------------------------------------------------
    ! Normalize results
    !-------------------------------------------------------------------
    if (valid_snapshots_count > 0) then
        delta_squared = delta_squared / real(valid_snapshots_count, 8)
        ! if (verbose) print *, "Normalization complete"
    else
        print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
        stop
    end if

    ! if (verbose) print *, "Subroutine compute_delta_squared completed successfully"
end subroutine shifted_msd

subroutine shifted_msd_beads(positions, delta_squared, verbose, nbeads, nsnapshots, natoms, M)
    implicit none

    ! Input arguments
    integer, intent(in) :: nbeads, nsnapshots, natoms, M
    logical, intent(in) :: verbose
    real(8), intent(in) :: positions(nbeads,nsnapshots, natoms, 3)  ! (time_steps, natoms, 3)

    ! Output argument
    real(8), intent(inout) :: delta_squared(M, natoms)  ! (M, natoms)

    ! Local variables
    integer :: snapshot_idx, atom_idx, ref_idx, b
    real(8), dimension(natoms, 3) :: ref_positions  ! Reference configuration
    real(8) :: displacement_squared  ! Temporary variable for squared displacement
    integer :: valid_snapshots_count  ! Counter for valid snapshots processed
    integer :: progress  ! Progress variables

    !-------------------------------------------------------------------
    ! Input validation
    !-------------------------------------------------------------------
    if (nbeads .le. 1 ) then
        print *, "Error: nbeads must be larger than 1"
        stop
    end if
    if (nsnapshots < 2 * M) then
        print *, "Error: nsnapshots must be at least 2*M"
        stop
    end if
    if (M <= 0) then
        print *, "Error: M must be greater than 0"
        stop
    end if
    if (natoms <= 0) then
        print *, "Error: natoms must be greater than 0"
        stop
    end if

    ! if (verbose) print *, "Input validation successful"

    !-------------------------------------------------------------------
    ! Initialization
    !-------------------------------------------------------------------
    delta_squared(:,:) = 0.0  ! Reset the output array
    valid_snapshots_count = 0  ! Initialize counter for valid snapshots
    progress = 0              ! Initialize progress counter

    ! if (verbose) print *, "Initialization complete"

    !-------------------------------------------------------------------
    ! Main computation loop
    !-------------------------------------------------------------------
    do b = 1, nbeads
        do ref_idx = 1, M
            ! Reference configuration at snapshot ref_idx
            ref_positions = positions(b,ref_idx, :, :)  

            do snapshot_idx = ref_idx, ref_idx + M - 1
                if (snapshot_idx > nsnapshots) exit  ! Prevent out-of-bounds access

                do atom_idx = 1, natoms
                    ! Compute squared displacement of atom atom_idx
                    displacement_squared = sum((positions(b,snapshot_idx, atom_idx, :) - ref_positions(atom_idx, :))**2)
                    delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
                        delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
                end do
            end do

            valid_snapshots_count = valid_snapshots_count + 1

            ! Update progress bar
            if (verbose) then
                progress = progress + 1
                if (progress == 1) then
                    ! For the first iteration, print progress and move to a new line
                    write(*, '(A)') ""  ! Newline after the first print
                    write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"                
                else
                    ! For subsequent iterations, update the progress on the same line
                    write(*, '(A)', advance='no') char(13)  ! Return carriage to overwrite the previous progress
                    write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"
                end if
            end if
        end do
    end do

    !-------------------------------------------------------------------
    ! Normalize results
    !-------------------------------------------------------------------
    if (valid_snapshots_count > 0) then
        delta_squared = delta_squared / real(valid_snapshots_count*nbeads, 8)
        ! if (verbose) print *, "Normalization complete"
    else
        print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
        stop
    end if

    ! if (verbose) print *, "Subroutine compute_delta_squared completed successfully"
end subroutine shifted_msd_beads

! subroutine shifted_msd_beads(positions, delta_squared, verbose, nbeads, nsnapshots, natoms, M)
!     implicit none

!     ! Input arguments
!     integer, intent(in) :: nbeads, nsnapshots, natoms, M
!     logical, intent(in) :: verbose
!     real(8), intent(in) :: positions(nbeads,nsnapshots, natoms, 3)  ! (time_steps, natoms, 3)

!     ! Output argument
!     real(8), intent(inout) :: delta_squared(M, natoms)  ! (M, natoms)

!     ! Local variables
!     integer :: snapshot_idx, atom_idx, ref_idx, b
!     real(8), dimension(natoms, 3) :: ref_positions  ! Reference configuration
!     real(8) :: displacement_squared  ! Temporary variable for squared displacement
!     integer :: valid_snapshots_count  ! Counter for valid snapshots processed
!     integer :: progress  ! Progress variables

!     !-------------------------------------------------------------------
!     ! Input validation
!     !-------------------------------------------------------------------
!     if (nbeads .le. 1) then
!         print *, "Error: nbeads must be larger than 1"
!         stop
!     end if
!     if (nsnapshots < 2 * M) then
!         print *, "Error: nsnapshots must be at least 2*M"
!         stop
!     end if
!     if (M <= 0) then
!         print *, "Error: M must be greater than 0"
!         stop
!     end if
!     if (natoms <= 0) then
!         print *, "Error: natoms must be greater than 0"
!         stop
!     end if

!     ! if (verbose) print *, "Input validation successful"

!     !-------------------------------------------------------------------
!     ! Initialization
!     !-------------------------------------------------------------------
!     delta_squared(:,:) = 0.0  ! Reset the output array
!     valid_snapshots_count = 0  ! Initialize counter for valid snapshots
!     progress = 0              ! Initialize progress counter

!     ! if (verbose) print *, "Initialization complete"

!     !-------------------------------------------------------------------
!     ! Main computation loop
!     !-------------------------------------------------------------------
!     ! do b = 1, nbeads
!     !     do ref_idx = 1, M
!     !         ! Reference configuration at snapshot ref_idx
!     !         ref_positions = positions(b,ref_idx, :, :)  

!     !         do snapshot_idx = ref_idx, ref_idx + M - 1
!     !             if (snapshot_idx > nsnapshots) exit  ! Prevent out-of-bounds access

!     !             do atom_idx = 1, natoms
!     !                 ! Compute squared displacement of atom atom_idx
!     !                 displacement_squared = sum((positions(b,snapshot_idx, atom_idx, :) - ref_positions(atom_idx, :))**2)
!     !                 delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
!     !                     delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
!     !             end do
!     !         end do

!     !         valid_snapshots_count = valid_snapshots_count + 1

!     !         ! Update progress bar
!     !         if (verbose) then
!     !             progress = progress + 1
!     !             if (progress == 1) then
!     !                 ! For the first iteration, print progress and move to a new line
!     !                 write(*, '(A)') ""  ! Newline after the first print
!     !                 write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / (M*nbeads), "%]"                
!     !             else
!     !                 ! For subsequent iterations, update the progress on the same line
!     !                 write(*, '(A)', advance='no') char(13)  ! Return carriage to overwrite the previous progress
!     !                 write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / (M*nbeads), "%]"
!     !             end if
!     !         end if
!     !     end do
!     ! end do

!     do ref_idx = 1, M
!         do snapshot_idx = ref_idx, ref_idx + M - 1
!             if (snapshot_idx > nsnapshots) exit
!             do atom_idx = 1, natoms
!                 do b = 1, nbeads
!                     ref_positions = positions(b, ref_idx, :, :)  ! careful, may need temp copy
!                     displacement_squared = sum((positions(b,snapshot_idx,atom_idx,:) - ref_positions(atom_idx,:))**2)
!                     delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
!                         delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared

!                     ! Update progress bar
!                     if (verbose) then
!                         progress = progress + 1
!                         if (progress == 1) then
!                             ! For the first iteration, print progress and move to a new line
!                             write(*, '(A)') ""  ! Newline after the first print
!                             write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / (M*nbeads), "%]"                
!                         else
!                             ! For subsequent iterations, update the progress on the same line
!                             write(*, '(A)', advance='no') char(13)  ! Return carriage to overwrite the previous progress
!                             write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / (M*nbeads), "%]"
!                         end if
!                     end if

!                 end do
!             end do
!         end do
!     end do

!     !-------------------------------------------------------------------
!     ! Normalize results
!     !-------------------------------------------------------------------
!     if (valid_snapshots_count > 0) then
!         delta_squared = delta_squared / real(valid_snapshots_count, 8) / nbeads
!         ! if (verbose) print *, "Normalization complete"
!     else
!         print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
!         stop
!     end if

!     ! if (verbose) print *, "Subroutine compute_delta_squared completed successfully"
! end subroutine shifted_msd_beads

!------------------------------------------------------------
! MPI-parallel version of shifted_msd_beads
! - Distribute beads (index b) across MPI ranks
! - Each rank computes local_delta and local_valid_count
! - Use MPI_Allreduce to sum across ranks
! - Assumes MPI_Init has been called externally
!------------------------------------------------------------
! subroutine shifted_msd_beads_mpi(positions, delta_squared, verbose, nbeads, nsnapshots, natoms, M)
!     use mpi
!     implicit none

!     ! Input arguments
!     integer, intent(in) :: nbeads, nsnapshots, natoms, M
!     logical, intent(in) :: verbose
!     real(8), intent(in) :: positions(nbeads,nsnapshots, natoms, 3)

!     ! In/out
!     real(8), intent(inout) :: delta_squared(M, natoms)

!     ! Local MPI variables
!     integer :: comm, ierr, rank, nprocs
!     integer :: i, b, ref_idx, snapshot_idx, atom_idx
!     integer :: b_start, b_end, beads_per_rank, remainder
!     real(8), allocatable :: local_delta(:,:)
!     real(8), dimension(:,:), allocatable :: ref_positions
!     real(8) :: displacement_squared
!     integer :: local_valid_count, global_valid_count
!     integer :: progress, total_work_local, work_done_local
!     integer(kind=MPI_ADDRESS_KIND) :: count  ! for MPI_Allreduce counts

!     ! For MPI_Allreduce on doubles, use MPI_DOUBLE_PRECISION (fortran constant)
!     comm = MPI_COMM_WORLD

!     call MPI_Init(ierr)

!     call MPI_Comm_rank(comm, rank, ierr)
!     call MPI_Comm_size(comm, nprocs, ierr)

!     !-- input validation (only on rank 0 print stops) -------------------
!     if (rank == 0) then
!         if (nbeads .le. 1) then
!             print *, "Error: nbeads must be larger than 1"
!             call MPI_Abort(comm, 1, ierr)
!         end if
!         if (nsnapshots < 2 * M) then
!             print *, "Error: nsnapshots must be at least 2*M"
!             call MPI_Abort(comm, 1, ierr)
!         end if
!         if (M <= 0) then
!             print *, "Error: M must be greater than 0"
!             call MPI_Abort(comm, 1, ierr)
!         end if
!         if (natoms <= 0) then
!             print *, "Error: natoms must be greater than 0"
!             call MPI_Abort(comm, 1, ierr)
!         end if
!     end if
!     call MPI_Barrier(comm, ierr)

!     !-------------------------------------------------------------------
!     ! Partition beads among ranks: simple block distribution
!     !-------------------------------------------------------------------
!     beads_per_rank = nbeads / nprocs
!     remainder = mod(nbeads, nprocs)
!     if (rank < remainder) then
!         beads_per_rank = beads_per_rank + 1
!         b_start = rank * beads_per_rank + 1
!     else
!         b_start = rank * beads_per_rank + remainder + 1
!     end if
!     b_end = b_start + beads_per_rank - 1
!     if (beads_per_rank == 0) then
!         b_start = 1
!         b_end = 0
!     end if

!     ! allocate local accumulation array
!     allocate(local_delta(M, natoms))
!     local_delta = 0.0_8

!     local_valid_count = 0
!     progress = 0

!     !-------------------------------------------------------------------
!     ! Main local computation: each rank loops over its beads only
!     !-------------------------------------------------------------------
!     do b = b_start, b_end
!         do ref_idx = 1, M
!             ! reference configuration for this bead and ref_idx:
!             ! extract into a temporary 2D array ref_positions(natoms,3)
!             allocate(ref_positions(natoms, 3))
!             ref_positions(:, :) = positions(b, ref_idx, :, :)

!             do snapshot_idx = ref_idx, ref_idx + M - 1
!                 if (snapshot_idx > nsnapshots) exit
!                 do atom_idx = 1, natoms
!                     displacement_squared = sum( ( positions(b, snapshot_idx, atom_idx, :) - ref_positions(atom_idx, :) )**2 )
!                     ! accumulate
!                     local_delta(snapshot_idx - ref_idx + 1, atom_idx) = &
!                         local_delta(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
!                 end do
!             end do

!             local_valid_count = local_valid_count + 1
!             deallocate(ref_positions)
!             ! progress counter (local)
!             if (verbose) then
!                 progress = progress + 1
!                 ! optional: print local progress per rank (not synchronized)
!                 write(*,'(A,I0,A,I0)') "Rank ", rank, ": processed ", progress, " refs"
!             end if
!         end do
!     end do

!     !-------------------------------------------------------------------
!     ! Reduce local_delta and local_valid_count across ranks
!     !-------------------------------------------------------------------
!     ! delta array has size M * natoms; perform Allreduce so every rank gets sum
!     count = int(M * natoms, kind=MPI_ADDRESS_KIND)
!     call MPI_Allreduce(local_delta, delta_squared, count, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)

!     call MPI_Allreduce(local_valid_count, global_valid_count, 1, MPI_INTEGER, MPI_SUM, comm, ierr)

!     !-------------------------------------------------------------------
!     ! Final normalization (same as original): divide by (global_valid_count * nbeads)
!     !-------------------------------------------------------------------
!     if (global_valid_count > 0) then
!         delta_squared = delta_squared / real(global_valid_count, 8) / real(nbeads, 8)
!     else
!         if (rank == 0) then
!             print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
!         end if
!         call MPI_Abort(comm, 2, ierr)
!     end if

!     ! deallocate
!     deallocate(local_delta)

!     call MPI_Finalize(ierr)

!     return
! end subroutine shifted_msd_beads_mpi
