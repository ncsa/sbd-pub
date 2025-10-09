/**
@file sbd/basic/out_of_place_func/davidson.h
@brief davidson diagonalization for ground state
*/
#ifndef SBD_BASIC_OUT_OF_PLACE_FUNC_DAVIDSON_H
#define SBD_BASIC_OUT_OF_PLACE_FUNC_DAVIDSON_H

namespace sbd {

  template <typename ElemT>
  void GetTotalD(const std::vector<ElemT> & hii,
		 std::vector<ElemT> & dii,
		 MPI_Comm h_comm) {
    size_t size_d = hii.size();
    dii.resize(static_cast<size_t>(size_d),ElemT(0.0));
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    MPI_Allreduce_c(hii.data(),dii.data(),size_d,DataT,MPI_SUM,h_comm);
  }

  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
		const std::vector<std::vector<std::vector<size_t>>> & ih,
		const std::vector<std::vector<std::vector<size_t>>> & jh,
		const std::vector<std::vector<std::vector<size_t>>> & tr,
		const std::vector<std::vector<std::vector<ElemT>>> & hij,
		std::vector<ElemT> & W,
		MPI_Comm & h_comm,
		MPI_Comm & b_comm,
		int max_iteration,
		int num_block,
		size_t bit_length,
		int data_width,
		RealT eps) {

    RealT eps_reg = 1.0e-12;
    
    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

    bool do_continue = true;

    for(int it=0; it < max_iteration; it++) {

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }
      
      for(int ib=0; ib < nb; ib++) {
	Zero(HC[ib]);
	mult(hii,ih,jh,tr,hij,C[ib],HC[ib],bit_length,data_width,h_comm,b_comm);
	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}
	
	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	// Get residual vector
	ElemT x = U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  W[is] = C[0][is] * x;
	}
	x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += C[kb][is] * x;
	  }
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] += E[0]*W[is];
	}

	// for stability
	MpiAllreduce(W,MPI_SUM,h_comm);
	MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*1.0));
#pragma	omp parallel for
        for(size_t is=0; is < W.size(); is++) {
          W[is] *= volp;
	}
#pragma omp parallel for
	for(size_t is=0; is < R.size(); is++) {
          R[is] *= volp;
	}

	RealT norm_W;
	Normalize(W,norm_W,b_comm);

	RealT norm_R;
	Normalize(R,norm_R,b_comm);

	for(int rank=0; rank < mpi_size_h; rank++) {
	  if( mpi_rank_b == 0 && mpi_rank_h == rank ) {
	    std::cout << " Davidson iteration " << it
		      << "." << ib
		      << " at h_comm rank " << mpi_rank_h
		      << ": (tol = " << norm_R << "):";
	    for(int p=0; p < std::min(ib+1,4); p++) {
	      std::cout << " " << E[p];
	    }
	    std::cout << std::endl;
	  }
	}

	

	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // Gram-Schmidt orthogonalization
	  for(int kb=0; kb < ib+1; kb++) {
	    ElemT olap;
	    InnerProduct(C[kb],C[ib+1],olap,b_comm);
	    olap *= ElemT(-1.0);
#pragma omp parallel for
	    for(size_t is=0; is < W.size(); is++) {
	      C[ib+1][is] += C[kb][is]*olap;
	    }
	  }

	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);
	  
	}
      }
      
      if( !do_continue ) {
	break;
      }

      // Restart with C[0] = W;
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }
      
    }

    free(H);
    free(U);
    free(E);
    
  } // end Davidson
		  
  
} // end namespace sbd

#endif // endif for SBD_HCBOSON_OUT_OF_PLACE_FUNC_DAVIDSON_H

