# Radia Perlman

## Early life
Perlman was born in 1951, Portsmouth, Virginia. She grew up in Loch Arbour, New Jersey. She is Jewish. Both of her parents worked as engineers for the US government. Her father worked on radar and her mother was a mathematician by training who worked as a computer programmer. During her school years Perlman found math and science to be “effortless and fascinating”, but had no problem achieving top grades in other subjects as well. She enjoyed playing the piano and French horn. While her mother helped her with her math homework, they mainly talked about literature and music. But she did not feel as though she fit underneath the stereotype of an "engineer", as she did not break apart computer parts.

Despite being the best science and math student in her school it was only when Perlman took a programming class in high school that she started to consider a career that involved computers. She was the only woman in the class and later reflected "I was not a hands-on type person. It never occurred to me to take anything apart. I assumed I'd either get electrocuted, or I'd break something". She graduated from Ocean Township High School in 1969.


## Education
As an undergraduate at MIT Perlman learned programming for a physics class. She was given her first paid job in 1971 as part-time programmer for the LOGO Lab at the (then) MIT Artificial Intelligence Laboratory, programming system software such as debuggers.

Working under the supervision of Seymour Papert, she developed a child-friendly version of the educational robotics language LOGO, called TORTIS ("Toddler's Own Recursive Turtle Interpreter System"). During research performed in 1974-76, young children – the youngest aged 3½ years – programmed a LOGO educational robot called a Turtle. Perlman has been described as a pioneer of teaching young children computer programming. Afterwards, she was inspired to make a new programming language that would teach much younger children similar to Logo, but using special "keyboards" and input devices. This project was abandoned because "being the only woman around, I wanted to be taken seriously as a 'scientist' and was a little embarrassed that my project involved cute little kids". MIT media project later tracked her down and told her that she started a new field called tangible user interface from the leftovers of her abandoned project. As a math grad at MIT she needed to find an adviser for her thesis, and joined the MIT group at BBN Technologies. There she first got involved with designing network protocols. Perlman obtained a B.S. and M.S. in Mathematics and a Ph.D. in Computer Science from MIT in 1988.  Her doctoral thesis on routing in environments where malicious network failures are present serves as the basis for much of the work that now exists in this area.

When studying at MIT in the late 60s she was one among the 50 or so women students, in a class of about 1,000 students. To begin with MIT only had one women’s dorm, limiting the number of women students that could study. When the men’s dorms at MIT became coed Perlman moved out of the women’s dorm into a mixed dorm, where she became the "resident female". She later said that she was so used to the gender imbalance, that it became normal. Only when she saw other women students among a crowd of men she noticed that "it kind of looked weird".


## Career
After graduation, she accepted a position with Bolt, Beranek, and Newman (BBN), a government contractor that developed software for network equipment. While working for BBN, Perlman made an impression on a manager for Digital Equipment Corp and was offered a job, joining the firm in 1980. During her time working at Digital, she quickly produced a solution that did exactly what the team wanted it to: the Spanning Tree Protocol. It allows a network to deliver data reliably by making it possible to design the network with redundant links. This setup provides automatic backup paths if an active link fails, and disables the links that are not part of the tree. This leaves a single, active path between any pair of network nodes. She is most famous for STP, which is fundamental to the operation of network bridges in many smaller networks. Perlman is the author of a textbook on networking called “Interconnections: Bridges, Routers, Switches, and Internetworking Protocols” and coauthor of another on network security called “Network Security: Private Communication in a Public World”, which is a now popular college textbook. Her contributions to network security include trust models for Public Key Infrastructure, data expiration, and distributed algorithms resilient despite malicious participants.

She left Digital in 1993 and joined Novell. Then, in 1997 she left Novell and joined Sun Microsystems. Over the course of her career, she has earned over 200 patents, 40 of them while working for Sun Microsystems, where in 2007 she held the title of Distinguished Engineer. She has taught courses at the University of Washington, Harvard University, MIT, and Texas A&M, and has been the keynote speaker at events all over the world. Perlman is the recipient of awards such as Lifetime Achievement awards from USENIX and the Association for Computing Machinery’s Special Interest Group on Data Communication (SIGCOMM).


### Spanning Tree Protocol
Perlman invented the spanning tree algorithm and protocol. While working as a consulting engineer at Digital Equipment Corporation (DEC) in 1984 she was tasked with developing a straightforward protocol that enabled network bridges to locate loops in a local area network (LAN). It was required that the protocol should use a constant amount of memory when implemented on the network devices, regardless of how large the network was. Building and expanding bridged networks was difficult because loops, where more than one path leads to the same destination, could result in the collapse of the network. Redundant paths in the network meant that a bridge could forward a frame in multiple directions. Therefore loops could cause Ethernet frames to fail to reach their destination, thus flooding the network. Perlman utilized the fact that bridges had unique 48 bit MAC addresses, and devised a network protocol so that bridges within the LAN communicated with one another. The algorithm implemented on all bridges in the network allowed the bridges to designate one root bridge in the network. Each bridge then mapped the network and determined the shortest path to the root bridge, deactivating other redundant paths. Despite Perlman's concerns that it took the Spanning Tree Protocol about a minute to react when changes in the network topology occurred, during which time a loop could bring down the network, it was standardized as 802.1d by the Institute of Electrical and Electronics Engineers (IEEE). Perlman said that the benefits of the protocol amount to the fact that "you don't have to worry about topology" when changing the way a LAN is connected. Perlman has, however, criticized changes that were made in the course of the standardization of the protocol.

Perlman published a poem on STP, called 'Algorhyme':

I think that I shall never see

A graph more lovely than a tree.



A tree whose crucial property

Is loop-free connectivity.



A tree which must be sure to span

So packets can reach every LAN.



First the root must be selected.

By ID it is elected.



Least cost paths from root are traced.

In the tree these paths are placed.



A mesh is made by folks like me

Then bridges find a spanning tree.

— Radia Perlman, Algorhyme


### Other network protocols
Perlman was the principal designer of the DECnet IV and V protocols, and IS-IS, the OSI equivalent of OSPF. She also made major contributions to the Connectionless Network Protocol (CLNP). Perlman has collaborated with Yakov Rekhter on developing network routing standards, such as the OSI Inter-Domain Routing Protocol (IDRP), the OSI equivalent of BGP. At DEC she also oversaw the transition from distance vector to link-state routing protocols. Link-state routing protocols had the advantage that they adapted to changes in the network topology faster, and DEC's link-state routing protocol was second only to the link-state routing protocol of the Advanced Research Projects Agency Network (ARPANET). While working on the DECnet project Perlman also helped to improve the intermediate-system to intermediate-system routing protocol, known as IS-IS, so that it could route the Internet Protocol (IP), AppleTalk and the Internetwork Packet Exchange (IPX) protocol. The Open Shortest Path First (OSPF) protocol relied in part on Perlman's research on fault-tolerant broadcasting of routing information.

Perlman subsequently worked as a network engineer for Sun Microsystems, now Oracle. She specialized in network and security protocols and, while working for Oracle, obtained more than 50 patents.

When standarizing her work on TRILL, a combined bridging and routing protocol that proposes to supersede STP, she included version 2 of the earlier "Algorhyme":

I hope that we shall one day see

A graph more lovely than a tree.



A graph to boost efficiency

While still configuration-free.



A network where RBridges can

Route packets to their target LAN.



The paths they find, to our elation,

Are least cost paths to destination!



With packet hop counts we now see,

The network need not be loop-free!



RBridges work transparently, 

Without a common spanning tree.

— Ray Perlman, Algorhyme V2, RFC 6325


## Awards
- Fellow of the Association for Computing Machinery, class of 2016[25]
- National Inventors Hall of Fame induction (2016)
- Internet Hall of Fame induction (2014)[26]
- SIGCOMM Award (2010)[27]
- IEEE Fellow in 2008 for contributions to network routing and security protocols[28]
- USENIX Lifetime Achievement Award (2006)
- Recipient of the first Anita Borg Institute Women of Vision Award for Innovation in 2005[29]
- Silicon Valley Intellectual Property Law Association Inventor of the year (2003)[30]
- Honorary Doctorate, Royal Institute of Technology (June 28, 2000)
- Twice named as one of the 20 most influential people in the industry by Data Communications magazine: in the 20th anniversary issue (January 15, 1992) and the 25th anniversary issue (January 15, 1997). Perlman is the only person to be named in both issues.