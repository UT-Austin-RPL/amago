---
layout: common
permalink: /
categories: projects
usemathjax: true
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents</title>

<!-- for mathjax support -->
{% if page.usemathjax %}
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
    });
  </script>
  <script type="text/javascript" async src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
{% endif %}


<!-- <meta property="og:image" content="src/figure/approach.png"> -->
<meta property="og:title" content="TRILL">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-5RB3JP5LNX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-5RB3JP5LNX');
</script>

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:20px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
    font-size:24px;
  }
  h3 {
    font-weight:300;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

	
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">


<style>
a {
  color: #bf5700;
  text-decoration: none;
  font-weight: 500;
}
</style>


<style>
highlight {
  color: #ff0000;
  text-decoration: none;
}
</style>

<div id="primarycontent">
<center><h1><strong>AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents</strong></h1></center>
<center><h2>
    <a href="https://jakegrigsby.github.io">Jake Grigsby<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://jimfan.me">Jim Fan<sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu<sup>1</sup></a>
  </h2>
  <h2>
    <a href="https://www.utexas.edu/"><sup>1</sup>The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.nvidia.com/en-us/research/"><sup>2</sup>NVIDIA Research</a>&nbsp;&nbsp;&nbsp;
  </h2>
<img src="./src/logos/amago_logo_2.png" alt="amagologo" width="250" align="center"/>
  <h2><a href="https://arxiv.org/abs/2310.09971v1">Paper</a>&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;<a href="https://github.com/UT-Austin-RPL/amago">Code</a></h2>
  </center>

 <center><p><span style="font-size:20px;"></span></p></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">

<h3>
"In-context" or "context-based" RL trains memory-equipped agents to identify and adapt to their new surroundings from test-time experience. In-context RL is extremely flexible because it turns meta-RL, zero-shot generalization, and long-term memory into the same problem. While this technique was one of the first approaches to deep meta-RL <a href="https://arxiv.org/abs/1611.02779">[1]</a>, it is often outperformed by more complicated methods. Fortunately, the right off-policy implementation details and tuning can make in-context RL stable and competitive <a href="https://arxiv.org/abs/2110.05038">[2]</a>. This creates a tradeoff: off-policy in-context RL is conceptually simple but hard to use and limits model size, memory length, and planning horizons. <b>AMAGO</b> redesigns off-policy sequence-based RL to break these bottlenecks and stably train long-context Transformers with end-to-end RL. AMAGO is open-source and designed to require minimal tuning with the goal of making in-context RL an easy-to-use default in new research on adaptive agents. <br><br>
</h3>

<hr>

<h1 align="center">Improving Off-Policy Actor-Critics with Transformers</h1>

<h3>
AMAGO improves memory and adaptation by optimizing long-context Transformers on sequences gathered from large off-policy datasets. This creates many technical challenges that we address with three main ideas:

<ol>
<li> <b> Sharing One Sequence Model. </b> &nbsp; AMAGO performs actor and critic updates in parallel on top of the outputs of a single sequence model that learns from every training objective and maximizes throughput. AMAGO's update looks more like supervised sequence modeling than an actor-critic. This approach is discouraged in previous work but can be stabilized with careful details. </li> <br>
<li> <b> Long-Horizon Off-Policy Updates. </b> &nbsp; AMAGO's learning update improves performance and reduces tuning by always giving the sequence model "something to learn about": we compute RL losses over many planning horizons (\(\gamma\)) that have different optimization landscapes depending on current performance. When all else fails, AMAGO includes an offline RL term that resembles supervised learning and does not depend on the scale of returns. This "multi-\(\gamma\)" update makes AMAGO especially effective for sparse rewards over long horizons.</li> <br>
<li> <b> Stabilizing Long-Context Transformers. </b> Both RL and Transformers can be unstable on their own, and combining them creates more obstacles. An especially relevant issue in memory-intensive RL is <i> attention entropy collapse</i>; the optimal memory patterns in RL environments can be far more specific than in language modeling. We use a stable Transformer block that prevents collapse and reduces tuning by letting us pick model sizes that are safely too large for the problem.</li>
</ol>
</h3>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr>
	<td align="center" valign="middle">
	  <a href="./src/figure/fig1_iclr_e_notation.pdf"><img src="./src/figure/fig1_iclr_e_notation.pdf" style="width:100%;"> </a>
        </td>
      </tr>
    </tbody>
  </table>

  <table align=center width=800px><tr><td><p align="justify" width="20%">


<br>
<h3>
In-Context RL's flexibility lets us evaluate AMAGO on many generalization, memory, and meta-learning domains.
</h3>

  </p></td></tr></table>

  
<br><br><hr> <h1 align="center">Meta-RL and Long-Term Memory</h1>


<h3>
Transformers are incredibly good at recall, and AMAGO lets us put that to effective use. We evaluate AMAGO on 39 environments from the <a href="https://arxiv.org/abs/2303.01859">POPGym suite</a>, where it leads to dramatic improvements in memory-intensive generalization problems and creates a strong default for sequence-based RL: <br><br>
</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody>
   <tr>
	<td align="center" valign="middle">
    </td>
	 <a href="./src/figure/popgym_summary_expanded_outliers.pdf"><img src="./src/figure/popgym_summary_expanded_outliers.pdf" style="width:100%;"> </a>
   </tr>
  </tbody>
</table>


<h3>
AMAGO treats meta-learning exactly the same as zero-shot generalization, and we demonstrate its stability and flexibility on several common meta-RL benchmarks. AMAGO also makes it easy to tune memory lengths to the adaptation difficulty of the problem, which can improve sample efficiency. <br> <br>
</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody>
   <tr>
	<td align="center" valign="middle">
    </td>
	 <a href="./src/figure/case_studies_arxiv_v2.pdf"><img src="./src/figure/case_studies_arxiv_v2.pdf" style="width:100%;"> </a>
   </tr>
  </tbody>
</table>


METAWORLD FIGURE HERE


<hr>

<h1 align="center">Adaptive Instruction-Following</h1>

<h3>
<b> An important benefit of off-policy learning is the ability to <i> relabel </i> rewards in hindsight </b>. AMAGO extends <a href="https://arxiv.org/abs/1707.01495">hindsight experience replay</a> (HER) to "instructions" or sequences of multiple goals. Relabeling instructions extends the diversity of our dataset and plays to the strengths of data-hungry Transformers while generating automatic exploration curricula for more complex objectives. The combination of AMAGO's relabeling, memory-based adaptation, and long-horizon learning update can be extremely effective in goal-conditioned generalization tasks. 

<br><br>

As an example, we evaluate instruction-conditioned agents in the procedurally generated worlds of <a href="https://arxiv.org/abs/2109.06780">Crafter</a>. Instructions are strings from a closed vocabulary of Crafter's achievement system, with added goals for navigation and block placement.
</h3>


<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody>
   <tr>
	<td align="center" valign="middle">
    </td>
	 <a href="./src/figure/crafter_condensed_results.pdf"><img src="./src/figure/crafter_condensed_results.pdf" style="width:100%;"> </a>
   </tr>
  </tbody>
</table>


<h3>
Above, we use several single-task instructions to evaluate the exploration capabilities of various ablations. As tasks require more exploration and adaptation to new world layouts, AMAGO's memory and relabeling become essential to success. Multi-step goals require considerable generalization, and AMAGO qualitatively demonstrates a clear understanding of the instruction with sample videos below. <br><br>
</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
   <tr>
    <td align="center" valign="middle">
"collect sapling, place plant x2, eat cow"
    </td>
    <td align="center" valign="middle">
"eat cow, make stone pickaxe, collect coal, make stone sword, defeat zombie"
    </td>
    <td align="center" valign="middle">
"make wood pickaxe, collect stone, build at (30, 30)"
    </td>
    <td align="center" valign="middle">
"travel to (10, 10), place stone, travel to (50, 50), place stone"
    </td>
   </tr>
   <tr> 
    <td align="center" valign="middle">
     <video muted controls width="200">
      <source src="./src/video/collect_sapling_place_plant_place_plant_eat_cow.mp4" type="video/mp4">
     </video>
    </td>
    <td align="center" valign="middle">
     <video muted controls width="200">
      <source src="./src/video/eat_cow_make_stone_pickaxe_collect_coal_make_stone_sword_defeat_zombie.mp4" type="video/mp4">
     </video>
    </td>
    <td align="center" valign="middle">
     <video muted controls width="200">
      <source src="./src/video/make_wood_pickaxe_collect_stone_build_30_30.mp4" type="video/mp4">
     </video>
    </td>
    <td align="center" valign="middle">
     <video muted controls width="200">
      <source src="./src/video/travel_10_10_place_stone_travel_50_50_place_stone.mp4" type="video/mp4">
     </video>
    </td>
   </tr>
</table>



<hr>

<h1 align="center">Using AMAGO</h1>

<h3>
In-context RL is applicable to any memory, generalization, or meta-learning problem, and we have designed AMAGO to be flexible enough to support all of those cases. Our code is fully open-source and includes examples of how to apply AMAGO to new domains. We hope our agent can serve as a strong baseline in the development of new benchmarks that require long-term memory and adaptation. <a href="https://github.com/UT-Austin-RPL/amago">Check it out on GitHub here</a>.
</h3>
 
<img src="./src/logos/amago_big_logo.png" alt="amagologo" width="200" align="center"/>
<hr>

<center><h1>Citation</h1></center>
<table align=center width=800px>
 <tr>
  <td>
  <pre><code style="display:block; overflow-x: auto">
@article{grigsby2023amago,
 title={AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents},
 author={Jake Grigsby and Linxi Fan and Yuke Zhu},
 year={2023},
 eprint={2310.09971},
 archivePrefix={arXiv},
 primaryClass={cs.LG}
}
  </code></pre>
  </td>
 </tr>
</table>
