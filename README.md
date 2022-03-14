# Source code for EquiDock: Independent SE(3)-Equivariant Models for End-to-End Rigid Protein Docking (ICLR 2022)

![EquiDock banner and concept](https://github.com/octavian-ganea/equidock_public/blob/main/equidock.png)


Please cite
```angular2html
@article{ganea2021independent,
  title={Independent SE (3)-Equivariant Models for End-to-End Rigid Protein Docking},
  author={Ganea, Octavian-Eugen and Huang, Xinyuan and Bunne, Charlotte and Bian, Yatao and Barzilay, Regina and Jaakkola, Tommi and Krause, Andreas},
  journal={arXiv preprint arXiv:2111.07786},
  year={2021}
}
```


## Dependencies
Current code works on Linux/Mac OSx only, you need to modify file paths to work on Windows.
```
python==3.9.10
numpy==1.22.1
cuda==10.1
torch==1.10.2
dgl==0.7.0
biopandas==0.2.8
ot==0.7.0
rdkit==2021.09.4
dgllife==0.2.8
joblib==1.1.0
```



## DB5.5 data

The raw DB5.5 dataset was already placed in the `data` directory from the original source:
```
https://zlab.umassmed.edu/benchmark/ or https://github.com/drorlab/DIPS
```
The raw pdb files of DB5.5 dataset are in the directory `./data/benchmark5.5/structures`

Then preprocess the raw data as follows to prepare data for rigid body docking:
```
# prepare data for rigid body docking
python preprocess_raw_data.py -n_jobs 40 -data db5 -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8
```

By default, `preprocess_raw_data.py` uses 10 neighbor for each node when constructing
the graph and uses only residues (coordinates being those of the alpha carbons). After running `preprocess_raw_data.py` you will get following 
ready-for-training data directory:

```
./cache/db5_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
```
with files
```
$ ls cache/db5_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
label_test.pkl			label_val.pkl			ligand_graph_train.bin		receptor_graph_test.bin		receptor_graph_val.bin
label_train.pkl			ligand_graph_test.bin		ligand_graph_val.bin		receptor_graph_train.bin
```


## DIPS data

Download the dataset (see `https://github.com/drorlab/DIPS` and `https://github.com/amorehead/DIPS-Plus`) :
```angular2html
mkdir -p ./DIPS/raw/pdb

rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ ./DIPS/raw/pdb
```

Follow the following first steps from `https://github.com/amorehead/DIPS-Plus` :
```angular2html
# Create data directories (if not already created):
mkdir project/datasets/DIPS/raw project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim project/datasets/DIPS/interim/external_feats project/datasets/DIPS/final project/datasets/DIPS/final/raw project/datasets/DIPS/final/processed

# Download the raw PDB files:
rsync -rlpt -v -z --delete --port=33444 --include='*.gz' --include='*.xz' --include='*/' --exclude '*' \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ project/datasets/DIPS/raw/pdb

# Extract the raw PDB files:
python3 project/datasets/builder/extract_raw_pdb_gz_archives.py project/datasets/DIPS/raw/pdb

# Process the raw PDB data into associated pair files:
python3 project/datasets/builder/make_dataset.py project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim --num_cpus 28 --source_type rcsb --bound

# Apply additional filtering criteria:
python3 project/datasets/builder/prune_pairs.py project/datasets/DIPS/interim/pairs project/datasets/DIPS/filters project/datasets/DIPS/interim/pairs-pruned --num_cpus 28
```

Then, place file `utils/partition_dips.py` in the `DIPS/src/` folder, use the `pairs-postprocessed-*.txt` files for the actual data splits used in our paper,
and run from the `DIPS/` folder the command: `python src/partition_dips.py  data/DIPS/interim/pairs-pruned/`. This creates the corresponding train/test/validation splits 
(again, using the exact splits in `pairs-postprocessed-*.txt`) of the 42K filtered pairs in DIPS. You should now have the following directory:

```angular2html
$ ls ./DIPS/data/DIPS/interim/pairs-pruned
0g  a6	ax  bo	cf  d6	dx  eo	ff  g6	gx  ho	if  j6	jx  ko	lf  m6	mx  no	of  p6				   pt  qk  rb  s2  st  tk  ub  v2  vt  wk  xb  y2  yt  zk
17  a7	ay  bp	cg  d7	dy  ep	fg  g7	gy  hp	ig  j7	jy  kp	lg  m7	my  np	og  p7				   pu  ql  rc  s3  su  tl  uc  v3  vu  wl  xc  y3  yu  zl
1a  a8	az  bq	ch  d8	dz  eq	fh  g8	gz  hq	ih  j8	jz  kq	lh  m8	mz  nq	oh  p8				   pv  qm  rd  s4  sv  tm  ud  v4  vv  wm  xd  y4  yv  zm
1b  a9	b0  br	ci  d9	e0  er	fi  g9	h0  hr	ii  j9	k0  kr	li  m9	n0  nr	oi  p9				   pw  qn  re  s5  sw  tn  ue  v5  vw  wn  xe  y5  yw  zn
1g  aa	b1  bs	cj  da	e1  es	fj  ga	h1  hs	ij  ja	k1  ks	lj  ma	n1  ns	oj  pa				   px  qo  rf  s6  sx  to  uf  v6  vx  wo  xf  y6  yx  zo
2a  ab	b2  bt	ck  db	e2  et	fk  gb	h2  ht	ik  jb	k2  kt	lk  mb	n2  nt	ok  pairs-postprocessed-test.txt   py  qp  rg  s7  sy  tp  ug  v7  vy  wp  xg  y7  yy  zp
2c  ac	b3  bu	cl  dc	e3  eu	fl  gc	h3  hu	il  jc	k3  ku	ll  mc	n3  nu	ol  pairs-postprocessed-train.txt  pz  qq  rh  s8  sz  tq  uh  v8  vz  wq  xh  y8  yz  zq
2e  ad	b4  bv	cm  dd	e4  ev	fm  gd	h4  hv	im  jd	k4  kv	lm  md	n4  nv	om  pairs-postprocessed.txt	   q0  qr  ri  s9  t0  tr  ui  v9  w0  wr  xi  y9  z0  zr
2g  ae	b5  bw	cn  de	e5  ew	fn  ge	h5  hw	in  je	k5  kw	ln  me	n5  nw	on  pairs-postprocessed-val.txt    q1  qs  rj  sa  t1  ts  uj  va  w1  ws  xj  ya  z1  zs
3c  af	b6  bx	co  df	e6  ex	fo  gf	h6  hx	io  jf	k6  kx	lo  mf	n6  nx	oo  pb				   q2  qt  rk  sb  t2  tt  uk  vb  w2  wt  xk  yb  z2  zt
3g  ag	b7  by	cp  dg	e7  ey	fp  gg	h7  hy	ip  jg	k7  ky	lp  mg	n7  ny	op  pc				   q3  qu  rl  sc  t3  tu  ul  vc  w3  wu  xl  yc  z3  zu
48  ah	b8  bz	cq  dh	e8  ez	fq  gh	h8  hz	iq  jh	k8  kz	lq  mh	n8  nz	oq  pd				   q4  qv  rm  sd  t4  tv  um  vd  w4  wv  xm  yd  z4  zv
4g  ai	b9  c0	cr  di	e9  f0	fr  gi	h9  i0	ir  ji	k9  l0	lr  mi	n9  o0	or  pe				   q5  qw  rn  se  t5  tw  un  ve  w5  ww  xn  ye  z5  zw
56  aj	ba  c1	cs  dj	ea  f1	fs  gj	ha  i1	is  jj	ka  l1	ls  mj	na  o1	os  pf				   q6  qx  ro  sf  t6  tx  uo  vf  w6  wx  xo  yf  z6  zx
5c  ak	bb  c2	ct  dk	eb  f2	ft  gk	hb  i2	it  jk	kb  l2	lt  mk	nb  o2	ot  pg				   q7  qy  rp  sg  t7  ty  up  vg  w7  wy  xp  yg  z7  zy
6g  al	bc  c3	cu  dl	ec  f3	fu  gl	hc  i3	iu  jl	kc  l3	lu  ml	nc  o3	ou  ph				   q8  qz  rq  sh  t8  tz  uq  vh  w8  wz  xq  yh  z8  zz
7g  am	bd  c4	cv  dm	ed  f4	fv  gm	hd  i4	iv  jm	kd  l4	lv  mm	nd  o4	ov  pi				   q9  r0  rr  si  t9  u0  ur  vi  w9  x0  xr  yi  z9
87  an	be  c5	cw  dn	ee  f5	fw  gn	he  i5	iw  jn	ke  l5	lw  mn	ne  o5	ow  pj				   qa  r1  rs  sj  ta  u1  us  vj  wa  x1  xs  yj  za
8g  ao	bf  c6	cx  do	ef  f6	fx  go	hf  i6	ix  jo	kf  l6	lx  mo	nf  o6	ox  pk				   qb  r2  rt  sk  tb  u2  ut  vk  wb  x2  xt  yk  zb
9g  ap	bg  c7	cy  dp	eg  f7	fy  gp	hg  i7	iy  jp	kg  l7	ly  mp	ng  o7	oy  pl				   qc  r3  ru  sl  tc  u3  uu  vl  wc  x3  xu  yl  zc
9h  aq	bh  c8	cz  dq	eh  f8	fz  gq	hh  i8	iz  jq	kh  l8	lz  mq	nh  o8	oz  pm				   qd  r4  rv  sm  td  u4  uv  vm  wd  x4  xv  ym  zd
a0  ar	bi  c9	d0  dr	ei  f9	g0  gr	hi  i9	j0  jr	ki  l9	m0  mr	ni  o9	p0  pn				   qe  r5  rw  sn  te  u5  uw  vn  we  x5  xw  yn  ze
a1  as	bj  ca	d1  ds	ej  fa	g1  gs	hj  ia	j1  js	kj  la	m1  ms	nj  oa	p1  po				   qf  r6  rx  so  tf  u6  ux  vo  wf  x6  xx  yo  zf
a2  at	bk  cb	d2  dt	ek  fb	g2  gt	hk  ib	j2  jt	kk  lb	m2  mt	nk  ob	p2  pp				   qg  r7  ry  sp  tg  u7  uy  vp  wg  x7  xy  yp  zg
a3  au	bl  cc	d3  du	el  fc	g3  gu	hl  ic	j3  ju	kl  lc	m3  mu	nl  oc	p3  pq				   qh  r8  rz  sq  th  u8  uz  vq  wh  x8  xz  yq  zh
a4  av	bm  cd	d4  dv	em  fd	g4  gv	hm  id	j4  jv	km  ld	m4  mv	nm  od	p4  pr				   qi  r9  s0  sr  ti  u9  v0  vr  wi  x9  y0  yr  zi
a5  aw	bn  ce	d5  dw	en  fe	g5  gw	hn  ie	j5  jw	kn  le	m5  mw	nn  oe	p5  ps				   qj  ra  s1  ss  tj  ua  v1  vs  wj  xa  y1  ys  zj
```

Then preprocess the raw data as follow to prepare data for rigid body docking:
```
# prepare data for rigid body docking
python preprocess_raw_data.py -n_jobs 60 -data dips -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8 -data_fraction 1.0
```

You should now obtain the following cache data directory:
```angular2html
$ ls cache/dips_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
label_test.pkl		     ligand_graph_val.bin		  receptor_graph_frac_1.0_train.bin
label_val.pkl		     ligand_graph_frac_1.0_train.bin  receptor_graph_test.bin
label_frac_1.0_train.pkl   ligand_graph_test.bin	      receptor_graph_val.bin
```



## Training
On GPU (works also on CPU, but it's very slow):
```angular2html
CUDA_VISIBLE_DEVICES=0 python -m src.train -hyper_search
```
or just specify your own params if you don't want to do hyperparam search. This will create checkpoints, tensorboard logs (you can visualize with tensorboard) and will store all stdout/stderr in a log file. This will train a model on DIPS first and, then, fine-tune it on DB5. Use `-toy` to train on DB5 only.

## Data splits
In our paper, we used the train/validation/test splits given by the files
```angular2html
DIPS: DIPS/data/DIPS/interim/pairs-pruned/pairs-postprocessed-*.txt
DB5: data/benchmark5.5/cv/cv_0/*.txt
```

## Inference

See `inference_rigid.py`.

## Pretrained models
Our paper pretrained models are available in folder `checkpts/`. By loading those (as in `inference_rigid.py`), you can also see 
which hyperparameters were used in those models (or directly from their names).

## Test and reproduce paper's numbers
Test sets used in our paper are given in `test_sets_pdb/`. Ground truth (bound) structures are in `test_sets_pdb/dips_test_random_transformed/complexes/`, 
while unbound structures (i.e., randomly rotated and translated ligands and receptors) are in `test_sets_pdb/dips_test_random_transformed/random_transformed/` 
and you should precisely use those for your predictions (or at least the ligands, while using the ground truth receptors like we do in `inference_rigid.py`). 
This test set was originally generated as a randomly sampled family-based subset of complexes in `./DIPS/data/DIPS/interim/pairs-pruned/pairs-postprocessed-test.txt`
using the file `src/test_all_methods/testset_random_transf.py`.


Run `python -m src.inference_rigid` to produce EquiDock's outputs for all test files. This will create a new directory of PDB output files in `test_sets_pdb/`. 

Get RMSD numbers from our papers using `python -m src.test_all_methods.eval_pdb_outputset`. You can use this script to evaluate all other baselines. Baselines' output PDB files are also provided in  `test_sets_pdb/`

### Note on steric clashes
Some clashes are possible in our model and we are working on mitigating this issue. Our current solution is a postprocessing clash removal step in `inference_rigid.py#L19`. Output files for DB5 are in `test_sets_pdb/db5_equidock_no_clashes_results/`. 




