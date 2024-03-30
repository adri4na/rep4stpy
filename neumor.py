import networkx as nx
from typing import Dict
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import random as rdd #para random.sample

from repast4py.network import write_network, read_network
from repast4py import context as ctx
from repast4py import core, random, schedule, logging, parameters

def not_metis(agts, n_ranks):
  n_ranks = 4
  max_inter_rank_connections = n_ranks -1

  disp= [i for i in range(0,n_ranks)]
  cruzan = [] 

  #agents_not_same_rank = [temp for temp in agts if temp[2] != 0]
  agents_not_same_rank = agts
  rdd.shuffle(agents_not_same_rank)

  for agent in agents_not_same_rank:
   random_element = rdd.sample(agents_not_same_rank, 1)
   if random_element[0][2] in disp:
     disp.remove(random_element[0][2])
     cruzan.append(random_element)
   else:
     if disp == []:
       break
     else:
       continue

  G = nx.Graph()

  for agent_id in agts:
          G.add_node(agent_id[0], type=agent_id[1], rank=agent_id[2])


  for agent in G.nodes():
      rank1 = G.nodes[agent]['rank']
      agents_in_same_rank = [agent_id2 for agent_id2, attr in G.nodes(data=True) if attr['rank'] == rank1]
    
      rdd.shuffle(agents_not_same_rank)
    

      agents_in_same_rank.remove(agent)  # Remove self-loop
      rdd.shuffle(agents_in_same_rank)  # Shuffle to randomize connections
      num_inter_rank_connections = 0

      for agent_id2 in agents_in_same_rank:
          if num_inter_rank_connections >= len(agents_in_same_rank)-1:
              break
          if not G.has_edge(agent, agent_id2):
              G.add_edge(agent, agent_id2)
              num_inter_rank_connections += 1

  elems0 = [agt for agt, attr in G.nodes(data=True) if attr['rank'] == 0]
  rand0 = rdd.sample(elems0, 1)

  for k  in cruzan:
  
    elems1 = [agt for agt, attr in G.nodes(data=True) if attr['rank'] == 1]
    for elem in elems1 :
      for cruz in cruzan:
        if elem == cruz[0][0]:
          cruzan.remove(cruz)
    
    rand1 = rdd.sample(elems1, 1) #del rank 1

    edge = [k[0][0], rand1[0]]
    rdd.shuffle(edge)
    G.add_edge(edge[0], edge[1])

  return(G)

def parse_edgestxt(txt):
    try:
        with open(txt, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise ValueError("No txt found [:p]")

    agents_ids = []

    with open(txt, 'r') as file:
        reading_edges = False
        for line in file:
            line = line.strip()
            if line.startswith("rumor_network"):
                drx = int(line[14])
                break

        if drx == 0 :
            G = nx.Graph()
        elif drx == 1:
            G = nx.DiGraph()
        else:
            raise ValueError("Invalid drx value in the file.")

        for line in file:
            line = line.strip()

            if line.startswith("rumor_network"):
                reading_agts = True
                continue

            if line == "EDGES":
                reading_edges = True
                reading_agts = False
                continue
            if reading_edges:
                source, target = map(int, line.split())
                G.add_edge(source, target)
            else:
                agents_ids.append(tuple(map(int, line.split())))

    return(G,list(G.edges()), agents_ids, drx)
    
def create_agent_network(edges,num_agents, num_ranks,drx):
    if drx == 0 :
        G = nx.Graph()
    elif drx == 1:
        G = nx.DiGraph()

    # Agregar nodos con atributos de tipo y rango
    for agent_id in num_agents:
        G.add_node(agent_id[0], type=agent_id[1], rank=agent_id[2])

    # Agregar conexiones entre agentes (enlace si pertenecen al mismo rango)
    for ed in edges:
        G.add_edge(ed[0], ed[1])

    return G

def plot_network_rank(G):
    fig = plt.figure(figsize=(120,120))
    # Crear un diccionario de colores para los rangos
    rank_colors = {0: 'yellow', 1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}

    # Obtener los colores de los nodos según el rango
    node_colors = [rank_colors[G.nodes[node]['rank']] for node in G.nodes]
    sorted_nodes = sorted(G.nodes())

    # Dibujar la red
    #pos = nx.spring_layout(G, seed=42)  # Posiciones de los nodos
    pos = nx.circular_layout(G)
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500)

    # Mostrar el gráfico
    plt.show()

def plot_agent_network(G):
    # Crear un diccionario de colores para los rangos
    rank_colors = {0: 'yellow', 1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5:'lime', 6:'cyan', 7:'orange', 8:'gray'}
    
    # Dividir los nodos en grupos según su rango
    rank_groups = {}
    for node in G.nodes():
        rank = G.nodes[node]['rank']
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(node)
    
    # Calcular la cantidad de nodos por grupo (cuarto del círculo)
    num_groups = len(rank_groups)
    nodes_per_group = len(G) // num_groups
    
    # Crear un diseño circular
    pos = nx.circular_layout(G)
    
    # Calcular el ángulo de inicio para cada grupo
    start_angles = np.linspace(0, 2 * np.pi, num_groups, endpoint=False)
    
    # Dibujar cada grupo en el diseño circular
    for i, nodes in enumerate(rank_groups.values()):
        start_angle = start_angles[i]
        end_angle = start_angle + (2 * np.pi) / num_groups
        for node in nodes:
            angle = np.linspace(start_angle, end_angle, len(nodes), endpoint=False)[nodes.index(node)]
            radius = 1.0  # Radio del círculo
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)
    
    # Obtener los colores de los nodos según el rango
    node_colors = [rank_colors[G.nodes[node]['rank']] for node in G.nodes]
    
    # Dibujar la red
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500)
    
    # Mostrar el gráfico
    plt.show()


def generate_network_file(fname: str, n_ranks: int, n_agents: int):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'rumor_network', fname, n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'rumor_network', fname, n_ranks)


model = None


class Neuron(core.Agent):

    TYPE = 0
    
    def __init__(self, nid: int, agent_type: int, rank: int, V = -0.06, g = 0.0, bglu = 0.0, gT = 0.0, Ie = 0.0, Isyn = 0.0, presyns = [], postsyns = []):
        super().__init__(nid, agent_type, rank)
        
        #########################
        ##PARAMETROS DEL AGENTE##
        ###self.received_rumor = received_rumor #SRC rumor
                
        #lista de pre-sinapticas cuando actua como post-sinaptica
        self.presyns = presyns
        
        #lista de neuronas de las que es pre sinaptica
        self.postsyns = postsyns
        
        #valores para inicializar el potencial de accion
        self.mode = 0.0 #inicializacion de variable mode para ser asignada en firing prob
        self.pFire = 0.0  
        self.Vresto = 0.0  
        self.Fthresho = 0.0  
        self.state = 0 # inicializacion de variable state para ser asignada en axn pot
        self.Vpahp = 0.0
        self.Isyn = Isyn
        self.Ie = Ie
        
        self.Tmin = 0.0  # theta min
        self.Tmax = 0.0
        self.Tmin2 = 0.0  # theta min 2
        self.Tmax2 = 0.0
        self.Thyper = -4  # timehyper

        self.beta = 20 #Licurgo ;no lineality constant
        self.kappa = 3
        self.a = 5
        self.r = 0
        self.bb = 2
        
        self.n_spikes = 0
        self.tp = 0.0 #spike probability
        		  #??
        self.V = V # en __init__
        self.C = 98.21 
        self.gL = 1
        self.RMP = self.V #Resting Membrane Potential 
        self.E_L = self.RMP

        self.W = 1
        self.En = 0.04 

        #conductance
        self.g = g # en __init__
        self.gT = gT  # en __init__
        self.gmax = 10 # r value
        self.f = 4.4 # change value
        self.T1 = 2 # r value
        self.T2 = 1 # r value
        self.tFire = -10

        self.Vpa = 0.04 
        self.AP_threshold = -0.05017 

        self.MTC = 42.78  #Membrane Time Constant
        self.Rm =  self.MTC / self.C

        self.AP_amp = 0.010307
        self.AP_haw = 0.0162

        self.Epahp = -0.060 
        self.tau = 100  
        self.fAHP_amplitud = -0.01220 

        self.vhyper = self.RMP

        self.IR = 173.26

        self.Tpost = 2
        
        #plasticidad
        self.Ipost = 0.0
        self.tgPA2 = 4000 #de bglu
        
        self.bglu = bglu #en __init__
        self.TnmdaF = 7
        self.TnmdaR = 1        

        self.tgPA = 4000
        

    def save(self):
    
        #self.state = [self.V, self.g, self.gT, self.bglu, self.Isyn, self.Ie, self.presyns, self.postsyns]
        
        return (self.uid, self.g)

    def update(self, data): #, data: list):
        
        #self.V = data[0]
        self.g = data
        #self.gT = data[2]
        #self.bglu = data[3]
        #self.Isyn = data[4]
        #self.Ie = data[5]
        #self.presyns = data[6]
        #self.postsyns = data[7]


def create_agent(nid, agent_type, rank, **kwargs):
    return Neuron(nid, agent_type, rank)


def restore_agent(agent_data):
    uid = agent_data[0]
    return Neuron(uid[0], uid[1], uid[2], agent_data[1]) #[1][0], agent_data[1][1], agent_data[1][2], agent_data[1][3], agent_data[1][4], agent_data[1][5], agent_data[1][6], agent_data[1][7])




class Model:

    def __init__(self, comm, params):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        
        self.comm = comm
        self.fpath = params['network_file']
        fpath = params['network_file']
        n_ranks = params['ranks']
        n_agents = params['num']
        #generate_network_file(fpath, n_ranks, n_agents)
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_agent, restore_agent)
        self.net = self.context.get_projection('rumor_network')

        self.G, self.edges, self.agts , self.drx= parse_edgestxt(fpath)
        #_, _, self.agts = parse_edgestxt(fpath)
        
        #self.G = not_metis(self.agts, n_ranks)
        #self.edges = self.G.edges
        
        self.rank = comm.Get_rank()
        
        #les da el valor inicial de peso a los edges
        for u, v in self.G.edges:
            self.G.edges[u, v]['weight'] = 30
            self.G.edges[u, v]['Isyn'] = 0
        
        self.voltajes_tick = []
        self.voltajes_total = []
        self.conductancia = []
        self.cond_agent = []
        self.corr_syn = []
        self.bglu = []
        self.ipost = []
        self.links_w = []
        
        self.frames_steps = []
        self.fig, self.ax = plt.subplots()
        
        self.dt = 0.5
        self.T_stim = 100
        
        self.E = 0.04  # potencial de equilibrio
        self.tpp = 12  	#plasltp
        self.npp = 500 	#plasltd  
        
        #para imprimir alguna prueba#
        self.idx = 0
        self.agt = 0
        ######################eliminar después 
        
        

        self.neuronas = []
        #inicia la funcion firing_probability de las neuronas 
        for neurona in self.context.agents(Neuron.TYPE):	
            self.firing_probability(neurona)
            self.neuronas.append(neurona)

        ###########################
        #slxn grupos 50 30 20 %       
        neuronas_n = len(self.neuronas)
        n50 = int(neuronas_n * 0.5) 
        n30 = int(neuronas_n * 0.3)
        n20 = neuronas_n - n50 - n30

        self.neuronas50 = rdd.sample(self.neuronas, n50)

        remaining_elements = [elem for elem in self.neuronas if elem not in self.neuronas50]
        self.neuronas30 = rdd.sample(remaining_elements, n30)

        self.neuronas20 = [elem for elem in remaining_elements if elem not in self.neuronas30]
        
        self.cruzan = []
        for u,v in self.edges:
        	for elem in self.agts:
                	if u == elem[0] :
                		uu = elem 
                	if v == elem[0] :
                		vv = elem
        	if uu[2] != vv[2] :		
                	self.cruzan.append([uu, vv])

        
        i=0
        #print(self.cruzan)
        #print(self.cruzaxxxn)
        self.uidx1 = (self.cruzan[i][0][0],self.cruzan[i][0][1],self.cruzan[i][0][2])
        self.uidx2 = (self.cruzan[i][1][0],self.cruzan[i][1][1],self.cruzan[i][1][2])
        self.d1v = []
        self.d1g = []
        self.d1gT = []
        self.d1ipost = []
        self.d1isyn = []
        self.d1bglu = []
        self.d2v = []
        self.d2gT = []
        self.d2g = []
        self.d2ipost = []
        self.d2isyn = []
        self.d2bglu = []
        self.dlvprom = []
        self.dlw = []
        
        self.rank1 = self.cruzan[i][0][2]
        self.rank2 = self.cruzan[i][1][2]
        ###########################
        ###########################
        #for n in (n50, n30,n20):
        #	self._seed_current(n, comm)


    def at_end(self):
        
        self.dd1v = MPI.COMM_WORLD.bcast(self.d1v, root= self.rank1)
        self.dd1g = MPI.COMM_WORLD.bcast(self.d1g, root= self.rank1) 
        self.dd1gT = MPI.COMM_WORLD.bcast(self.d1gT, root= self.rank1)
        self.dd1ipost = MPI.COMM_WORLD.bcast(self.d1ipost, root= self.rank1)
        self.dd1isyn = MPI.COMM_WORLD.bcast(self.d1isyn , root= self.rank1)
        self.dd1bglu  = MPI.COMM_WORLD.bcast(self.d1bglu , root= self.rank1)
        self.dd2v = MPI.COMM_WORLD.bcast(self.d2v , root= self.rank2)
        self.dd2gT = MPI.COMM_WORLD.bcast(self.d2gT , root=self.rank2)
        self.dd2g = MPI.COMM_WORLD.bcast(self.d2g , root=self.rank2)
        self.dd2ipost = MPI.COMM_WORLD.bcast(self.d2ipost , root=self.rank2)
        self.dd2isyn = MPI.COMM_WORLD.bcast(self.d2isyn , root=self.rank2)
        self.dd2bglu = MPI.COMM_WORLD.bcast(self.d2bglu, root=self.rank2)
        
        #print(self.G.edges(data=True))
        self.graf_iv()
        
        
        if self.rank == 0:
        	#_, edges, agts = parse_edgestxt(self.fpath)

        	agent_network = create_agent_network(self.edges,self.agts, 4, self.drx)

        	plot_agent_network(agent_network)
        

    def step(self):

        rng = random.default_rng
        
        voltajes_tick = []
        gs = []
        Isyn = []
        g_unico = []
        pre_syns = []
        ipost_tick = []
        bglu_tick = [] 
        w_tick = []
        
        tick = self.runner.schedule.tick   

        if tick == 10 :#and self.rank== self.rank1:
            
            #neuron = self.context.agent(self.uidx1)
            #neuron.Ie = 10
            
            #for uid in self.neuronas50:
            	#uid = (i, 0 , self.rank)
            	#neuron = self.context.agent(uid)
        #    	neuron.tgPA = 0
            #	uid.Ie = 10

            for neuron in self.context.agents(Neuron.TYPE):
            	neuron.Ie = 10

        if tick == 400 :#and self.rank==self.rank1:
            for uid in self.neuronas50:
            	#uid = (i, 0 , self.rank)
            	#neuron = self.context.agent(uid)
        #    	neuron.tgPA = 0
            	uid.Ie = 0

            #for neuron in self.context.agents(Neuron.TYPE):
            #	neuron.Ie = 0

        if tick == 20 :#and self.rank==self.rank1:
            for neuron in self.context.agents():
            	neuron.Ie = 10
        
        for agent in self.context.agents():            
            
            spikes_at = agent.n_spikes
            
            #almacenamiento
            voltajes_tick.append(agent.V)            
            ipost_tick.append(agent.Ipost)
            bglu_tick.append(agent.bglu)
            g_unico.append(agent.g)           
            gs.append(agent.gT) #apendiza los g's para la grafica
            Isyn.append(agent.Isyn)# " " isyn " "
            
            g_total = 0 
            
            ##voltaje
            agent.Vpahp += self.dt * ( agent.tp * agent.fAHP_amplitud - agent.Vpahp) / agent.tau
                          
            agent.V += (1/ agent.C) * ( ((agent.Isyn + agent.Ie)* 0.001 - agent.gL * (agent.V - agent.E_L)) + agent.Vpahp * (agent.Epahp - agent.V))
            self.action_potential(agent)
            ##
            
            ##################
            ##conductancia
            agent.g = agent.gmax * (np.exp(-1*agent.tgPA/agent.T1) - np.exp(-1*agent.tgPA/agent.T2))
            ##################
            agent.tgPA2 += 1
            
            ##Ipost
            agent.Ipost = 0.5 * agent.tgPA * np.exp( 1 - 0.5* agent.tgPA)
            ##
            if agent.tgPA == 3:
                agent.tgPA2 = 1
            
            ##bglu
            agent.bglu = np.exp(- agent.tgPA2/agent.TnmdaF) * ( 1 - np.exp( - agent.tgPA2 / agent.TnmdaR) )
            
            nghs = []
            tgpa_temp = 0
            
            
            for elem in self.cruzan :
            	if agent.uid == elem[0] :
            		self.comm.send([agent.g, agent.bglu], dest =  elem[1][2], tag= elem[0][0])
            
        #self.comm.Barrier()
        
        for agent in self.context.agents():  
            for ngh in self.net.graph.predecessors(agent):
                if ngh.local_rank == self.rank : 
                	g_total += ngh.g
                	nghs.append(ngh)
                if ngh.local_rank != self.rank:
                	props = self.comm.recv(source = ngh.uid[2], tag = ngh.uid[0])
                	g_total += props[0]
                	print("---", props)
            
            agent.gT = g_total
            agent.Isyn = agent.gT * (agent.En - agent.V)
            
            v = agent.uid[0]
            for (u, w) in self.G.edges :
            	if (u,v)==(u,w):
            		for elem in nghs:
            			if int(u) == elem.uid[0] :
            				upre = elem
            		
            				dw = self.dwdt(upre.bglu, agent.Ipost, self.G[u][v]['weight'])
            				self.G[u][v]['weight'] +=  0.5*dw*0.25
            				self.G[u][v]['Isyn'] = self.Isynaptic(upre.g, agent.V)
             
            if agent.uid == self.uidx1 :
                self.d1v.append(agent.V)
                self.d1g.append(agent.g)
                self.d1bglu.append(agent.bglu)
                self.d1ipost.append(agent.Ipost)

            if agent.uid == self.uidx2 :
                self.d2v.append(agent.V)
                self.d2g.append(agent.g)
                self.d2bglu.append(agent.bglu)
                self.d2ipost.append(agent.Ipost)        

            if agent.uid == self.uidx1 :
                self.d1gT.append(agent.gT)
                self.d1isyn.append(agent.Isyn)

            if agent.uid == self.uidx2 :
                self.d2gT.append(agent.gT)
                self.d2isyn.append(agent.Isyn) 
            
       
        #print(self.G.edges(data=True))
        
        self.voltajes_total.append(voltajes_tick)
        self.conductancia.append(gs)
        self.cond_agent.append(g_unico)
        self.corr_syn.append(Isyn)
        self.ipost.append(ipost_tick)
        self.bglu.append(bglu_tick)
        
        #self.frames_steps.append([self.G])

        #self.context.synchronize(restore_agent)
        #print("x ", tick)

    def start(self):
        self.runner.execute()



    def firing_probability(self, neuron):
        neuron.RMP = neuron.E_L #resting membrane potential¿¿
        neuron.Tmin = neuron.RMP 
        neuron.Tmax = neuron.AP_threshold 
        neuron.Tmin2 = neuron.RMP
        neuron.Tmax2 = neuron.AP_threshold

        neuron.mode = 1 /(1 +((10 ** neuron.kappa) / (10 **(neuron.a * neuron.r))) ** neuron.bb)
        
        neuron.Vresto = neuron.Tmin + (neuron.Tmin2 - neuron.Tmin) * neuron.mode
        neuron.Fthresho = neuron.Tmax + (neuron.Tmax2 - neuron.Tmax) * neuron.mode

        neuron.pFire = (neuron.V - neuron.Vresto) / (neuron.Fthresho - neuron.Vresto)
        neuron.vhyper = neuron.RMP

    def action_potential(self, neuron):
    	if neuron.state == 2 :
            neuron.state = 0
            neuron.tp = 0
            
    	if neuron.state == 1 :
            neuron.V = neuron.vhyper - 0.003
            #neuron.tgPA += 1
            neuron.tgPA += self.dt
            neuron.state = 2 
            neuron.tFire = self.runner.schedule.tick
    	
    	if neuron.state == 0 :
            neuron.mode = 1 / (1 + (10**neuron.kappa)/(10**(neuron.a * neuron.r)) )**neuron.bb
            neuron.pFire = (neuron.V- neuron.Vresto)/(neuron.Fthresho - neuron.Vresto)
            if neuron.pFire < 0:
    	        neuron.pFire = 0
            if neuron.pFire > 1:
    	        neuron.pFire = 1
            if (self.runner.schedule.tick - neuron.tFire < 4) and (self.runner.schedule.tick - neuron.tFire >= 0):
    	        neuron.pFire = 0
            neuron.pFire = neuron.pFire ** neuron.beta

            rng = np.random.default_rng()
            randomPA = rng.random()
            if randomPA <= (neuron.pFire * self.dt) :
    	        neuron.tgPA = 0
    	        neuron.V = 0.04
    	        neuron.state = 1
    	        neuron.tp = 1
    	        neuron.n_spikes += 1
            else :   
    	        #neuron.tgPA += 1  
    	        neuron.tgPA += self.dt
    	            
    def dwdt(self, e1bglu, e2Ipost, weight):
        #e1bglu = presyn.bglu
        #e2Ipost = postsyn.Ipost
        dw = (41.8 - weight) * e1bglu* ( e2Ipost / self.tpp ) + ( 12.25 - weight) * ( e1bglu + e2Ipost) / self.npp
        return(dw)

    def Isynaptic(self, presyng, postsynV):
        g = presyng 
        V = postsynV
        Isyn = g * (self.E - V)
        return(Isyn) 
        
    def _seed_current(self, init_neurons_w_current: int, comm):
        world_size = comm.Get_size() #nro de ranks
        # np array of world size, the value of i'th element of the array is the number of rumors to seed on rank i.
        nr_counts = np.zeros(world_size, np.int32) #por ejm: si n_ranks = 4 -> rumor_counts = [0 0 0 0]
        if (self.rank == 0):
            for _ in range(init_neurons_w_current):
                idx = random.default_rng.integers(0, high=world_size)
                nr_counts[idx] += 1
                self.idx = idx
                
        
        nr_count = np.empty(1, dtype=np.int32)
        comm.Scatter(nr_counts, nr_count, root=0)
        #envia desde el rank 0, el nro de neuronas con corriente (de todos los ranks) a la variable nrcount del rank local

        for agent in self.context.agents(count=nr_count[0], shuffle=True):
            agent.Ie = 10

    def graf_iv(self):    

        ntcks = len(self.voltajes_total)
        
        self.total_neuron_rank = len(self.neuronas)
        
        ####
        
        t = list(range(1, ntcks+1))
        
        datos_plot = []
        datos_plotc = []
        datos_plotcunico = []
        datos_plotisyn = []
        datos_plotipost = []
        datos_plotbglu = []
        
        
        for i in range(self.total_neuron_rank):
            datos_plot.append([])
            datos_plotc.append([])
            datos_plotcunico.append([])
            datos_plotisyn.append([])
            datos_plotipost.append([])   
            datos_plotbglu.append([])
     
        
        for i in range(ntcks):
            temp_vtick = self.voltajes_total[i]
            for j in range(self.total_neuron_rank):
    	        datos_plot[j].append(temp_vtick[j])

        for i in range(ntcks):
            temp_ctick = self.conductancia[i]        
            for j in range(self.total_neuron_rank):
    	        datos_plotc[j].append(temp_ctick[j])

        for i in range(ntcks):
            temp_ctick = self.cond_agent[i]        
            for j in range(self.total_neuron_rank):
    	        datos_plotcunico[j].append(temp_ctick[j])
        
        for i in range(ntcks):
            temp_itick = self.corr_syn[i]        
            for j in range(self.total_neuron_rank):
    	        datos_plotisyn[j].append(temp_itick[j])
        
        for i in range(ntcks):
            temp_itick = self.ipost[i]        
            for j in range(self.total_neuron_rank):
    	        datos_plotipost[j].append(temp_itick[j])
    	        
        for i in range(ntcks):
            temp_btick = self.bglu[i]        
            for j in range(self.total_neuron_rank):
    	        datos_plotbglu[j].append(temp_btick[j])
 
        
        plt.plot(t,self.dd1v, label = 'N# '+str(self.uidx1))
        plt.plot(t,self.dd2v, label = 'N# '+str(self.uidx2))

        plt.ylabel('V')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank)+ 'Neuronas'+ str(self.uidx1)+ '&' + str(self.uidx2))   
        plt.legend()
        plt.show()
        
        plt.plot(t,self.dd1g, label = 'N# '+str(self.uidx1))
        plt.plot(t,self.dd2g, label = 'N# '+str(self.uidx2))

        plt.ylabel('g')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank)+ 'Neuronas'+ str(self.uidx1)+ '&' + str(self.uidx2))   
        plt.legend()
        plt.show()
            
        plt.plot(t,self.dd1ipost, label = 'N# '+str(self.uidx1))
        plt.plot(t,self.dd2ipost, label = 'N# '+str(self.uidx2))

        plt.ylabel('Ipost')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank)+ 'Neuronas'+ str(self.uidx1)+ '&' + str(self.uidx2))   
        plt.legend()
        plt.show()
        
        plt.plot(t,self.dd1isyn, label = 'N# '+str(self.uidx1))
        plt.plot(t,self.dd2isyn, label = 'N# '+str(self.uidx2))

        plt.ylabel('Ipost')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank)+ 'Neuronas'+ str(self.uidx1)+ '&' + str(self.uidx2))   
        plt.legend()
        plt.show()
            
        plt.plot(t,self.dd1bglu, label = 'N# '+str(self.uidx1))
        plt.plot(t,self.dd2bglu, label = 'N# '+str(self.uidx2))

        plt.ylabel('bglu')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank)+ 'Neuronas'+ str(self.uidx1)+ '&' + str(self.uidx2))   
        plt.legend()
        plt.show()
            

        for k in range(self.total_neuron_rank):
            if k % 1 == 0:
    	        plt.plot(t,datos_plot[k], label = 'N#'+str(k+1)+' spikes ='+str(self.neuronas[k].n_spikes))
    	        #plt.plot(t,datos_plotcunico[k],'--', label = 'N#'+str(k+1)+'COND AGT'+str(self.neuronas[k])+str(self.rank))  

        plt.ylabel('V(t) ')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank))   
        plt.legend()
        plt.show()



        for k in range(self.total_neuron_rank):
            if k % 1 == 0:
    	        plt.plot(t,datos_plotipost[k], label = 'N#'+str(k+1)+' Ipost'+str(self.neuronas[k]))

        plt.ylabel('Ipost(t) ')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank))   
        plt.legend()
        plt.show()


        for k in range(self.total_neuron_rank):
            if k % 1 == 0:
    	        plt.plot(t,datos_plotbglu[k],'--', label = 'N#'+str(k+1)+'bglu '+str(self.neuronas[k])+str(self.rank))  

        plt.ylabel('bglu(t) ')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank))   
        plt.legend()
        plt.show()

        
        for k in range(self.total_neuron_rank):
            if k % 1 == 0:
    	        #plt.plot(t,datos_plot[k], label = 'N#'+str(k+1)+' spikes ='+str(self.neuronas[k].n_spikes))
    	        plt.plot(t,datos_plotcunico[k],'--', label = 'N#'+str(k+1)+'COND AGT'+str(self.neuronas[k])+str(self.rank))  

        plt.ylabel('g')
        plt.xlabel('Ticks')  
        plt.title('RANK'+ str(self.rank))   
        plt.legend()
        plt.show()

        #for k in range(self.total_neuron_rank):
        #    if k % 1 == 0:
    	#        plt.plot(t,datos_plotc[k], label = 'N#'+str(k+1)+'COND TOTAL'+str(self.neuronas[k])+str(self.rank))
    	        #plt.plot(t,datos_plotcunico[k],'--', label = 'N#'+str(k+1)+'COND AGT'+str(self.neuronas[k])+str(self.rank))  
    	        

        #plt.ylabel('g(t) ')
        #plt.xlabel('Ticks')
        #plt.title('RANK'+ str(self.rank))   
        #plt.legend()
        #plt.show()
        
        
        for k in range(self.total_neuron_rank):
            if k % 1 == 0:
    	        plt.plot(t,datos_plotisyn[k], label = 'N#'+str(k+1)+'Isyn '+str(self.neuronas[k])+str(self.rank))
    	        

        plt.ylabel('Isyn(t) ')
        plt.xlabel('Ticks')
        plt.title('RANK'+ str(self.rank))   
        plt.legend()
        plt.show()
        
        #for k in range(self.total_neuron_rank):
        #    if k % 4 == 0:
    	#        plt.plot(t,datos_plotcunico[k], label = 'N#'+str(k+1)+'COND AGT')
    	           
        #plt.ylabel('V(t) ')
        #plt.xlabel('Ticks')   
        #plt.legend()
        #plt.show()

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
