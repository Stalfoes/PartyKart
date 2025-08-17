# Instruction Manual

## Bracket Generation
This is the core functionality of the Beerio Kart program. This program takes in a configuration dictionary and produces a list of races that aim to:
1. Minimize the number of total races
1. Ensure that every player races the same number of times
1. Ensure races are at least size 3, and at most size 4
1. Increase the overall fairness by:
    1. Ensuring every player plays against every other player an equal number of times
    2. Ensuring every player has their races spaced out as much as possible and as fairly as possible

Obviously, this is a difficult problem to solve. If you'd like to read more about the problem and how I aimed to solve it, please read the section on [Technical Implementation Details](#technical-implementation-details).

### How to use the Bracket Generator
1. Ensure you're in the `RonyPartyKart` directory in any terminal you plan to run the program in. In VisualStudioCode, which is where I typically run things, this is the bottommost middle part of the screen that is black with white text.
1. Edit the `config.yaml` file to your specifications. I believe the file to be fairly straightforward, but in case it isn't, here's a rundown:
> - The `roster` section should outline ALL the players. Including ones that have left! If you wish to add a new player to the night that wasn't in some of the previous races, put their name here, following the format of the other players (TAB + - + [their name]).
> - The `people_that_left` section outlines players that have left after playing some/no races. This is typically `null`, indicating that nobody has left. However, if you want to add someone, delete the text saying `null`, press ENTER/RETURN and add their name just like in the `roster` section (TAB + - + [their name]).
> - The `num_races_each` section specifies the exact number of races you want everybody in the `roster` (and not in `people_that_left`) to play. Just change the number to what you want.
> - The `previous_races` section outlines all the races that have already happened, and need to be kept the same (constant). These are a constraint to the program. The more of these you add, the harder it really is for the program to find a good solution. However, if someone leaves, you have to add them unless you want to restart the whole night. This is typically `null`, indicating it's a fresh night. If you wish to add some of these races, delete the text saying `null`, press ENTER/RETURN and press TAB, then -, then a space and type each name (EXACTLY HOW IT IS IN THE ROSTER) with spaces separating them.
> - The `algorithmic` section need not be touched! This is technical and affects how the program finds solutions. Including the amount of randomness, how much of the computer's processor to use, the random seed to start with, and how long to look for a solution.
3. Type `python3 run.py` in the terminal, and press ENTER/RETURN to run the program! It will show some progress bars and output some things you need not care about. When it finishes, it will open up a new window showing the new races. If you don't wanna stare at it, just close this. You can move onto running the application.

## The Application
The application is just a little app I wrote to make things pretty and keep track of racers' scores and their positions on the leaderboard and display the races and allow you to input how they did so you can update the leaderboard. There's just a few keyboard presses that control the whole thing (_I wrote it to be simple, not amazingly intuitive or pretty_).
### How to use the Application
1. Like the steps above for the Bracket Generation, ensure you're in the `RonyPartyKart` directory in the terminal.
2. Run `python3 application/run.py` to run the application. A window should pop up.
3. This window uses the races that the Bracket Generation program outputted as well as whatever information you put into that silly little `config.yaml` file. It should load up to whatever race after the last one you specified in `config.yaml`'s `previous_races` section. We assume these already happened and their order/standings was preserved and already update leaderboards based on that.
4. If you ever wish to go back in the program (because you made a mistake) just press BACKSPACE/DELETE. This rolls the program back one race and the leaderboard accordingly.
5. When you have a current race displayed on the left side of the screen, you'll see 3-4 racers names. You can play that race on your Mario Kart. When that race completes, hover over a name, and press keys 1,2,3, or 4 on the keyboard to send that player's name to that position in the Current Race. When you've ordered everyone, press SPACE to update the scoreboard and move onto the next race.
6. If at any point you wish to save where you are in this application's execution, press S.
7. If at any point you wish to load the previous save, press L.

## Adding and Removing a Player
So, someone left or joined your tournament late after you've already played some races. That fucker. Well, good news, we thought of that.
1. **Do not close down the application just yet!**
1. Press `S` on your keyboard on the application window.
1. Locate the terminal in which you ran the application from. It should now have some text in it surrounded by some lines. Go ahead and press CTRL+C or Command+C on that to copy it. Don't copy the lines, just the text and the `-` in front of those list of names. These are the races you've already played. We need to keep these for a few seconds.
1. If you get confused in any of the following steps, I already kinda discussed them above in the [How to use the Bracket Generator](#how-to-use-the-bracket-generator) section.
1. Locate the `config.yaml` file. Within the file, locate the `previous_races` section. If there's a `null` beside it, delete it and press ENTER/RETURN.
1. Paste (CTRL+V or Command+V) those lines under the `previous_races:` header. Select all of what you just pased, and hit TAB once on the keyboard. Ensure it's just tabbed once more than the `previous_races:` header above it.
1. **If someone left...** add their name to the `people_that_left:` header by deleting the `null` text if there is one, and adding their name EXACTLY like it appears in the `roster:` section above. So, we delete `null` (if there is), hit ENTER/RETURN, hit TAB, press `-`, then a SPACE, then put their name there. DO NOT delete their name from the roster!
1. **If someone joiend late...** we need to add their name to the `roster:` section just like everyone else's name appears.
1. You may now close down the application.
1. Now, run the Bracket Generator program using `python3 run.py`, and you should see a bunch of progress bars. Wait for it to finish and open a window. You may close that down.
1. Go ahead and re-run the application. It should now have new brackets generated without the person that left or with the new person. It should load up exactly where you left off with the leaderboards being the same. Just with all new races coming up and currently.

# Technical Implementation Details
Nerd. Looking at technical implementation details 🤓.
## Implementation of Bracket Generation
I'll start off by defining a few variables I use for terminology:
- $V$ is the length of the roster. i.e. the total number of people.
- $R$ is the amount of times each player should appear in all the races. We want this to be the exact same for every player. No exceptions.
- $k$ is the size of the races. In typical [Block Design](https://en.wikipedia.org/wiki/Block_design) this is a hard constraint, and a single integer. In our case, races can be size 3 or 4. So $k\in\{3,4\}$.
- The list of races, or a race, is _invalid_ if it contains a racer's name more than once. For example, if a race is $\{$ John, Alex, Mark, John $\}$, then it's invalid because John appears twice. John cannot race against themself.
- We define a _fitness_ function as a function that takes in the list of races and outputs a single floating point number on how good those list of races are based on our goals. The fitness for an invalid set of races or a race is $-\infty$.
- _Evenness_ is a fitness function defined to help the program know how fair or unfair its races are by determining how evenly people play against all other people. For example, if John is a terrible MarioKarter and Alex is amazing, it would be very unfair if Mark played against either one of them much more than the other. If Mark played against John the entire night, his score would be much more inflated than anyone else's; and on the flip side, if Mark played against Alex the entire night then his score wouldn't be as high as if it were more fair. I'll get into how this is calculated later. 


To minimize the number of total races, we want to keep $k=4$ as much as possible, and only choose $k=3$ when absolutely necessary. The necessary conditions for that would be if we had $R$ and $V$ such that it's  not divisible by $4$. So we shift some people maybe from a race of $4$ to form races that are at least size $3$ instead of $1$ or $2$.

To ensure $R$ is kept constant for everyone, I start by randomly generating a list of names that is size $R\cdot V$, so that every person in the roster has their name appear $R$ times, and I then shuffle that list of names. I then section those names sequentially into groups of size 4 or 3. I maximize the groups of size 4 and only shift people if a 3 is necessary. Of course, when I do this shuffling, the races may be invalid. Ignore this for now, we'll touch on it later on its ramifications and how to mitigate this.

We start by counting how many times a pair of people occurs in all the races. For example, we count how many times John races against Alex by counting the pair $($ John, Alex $)$. We even count pairs where a racer races against themselves, e.g. $($ Alex, Alex $)$ (remember this makes races invalid). I'll define the set of all pairs as $\mathcal{P}$.

We then take note that we can actually swap racers between races. For example, we can move Alex to a race where he is not and in exchange take John from that race and move him to the race we just took Alex from. We define a swap as a pair of pairs $\mathcal{S}\overset{.}{=}\mathcal{P}\times\mathcal{P}$. For a swap to be any bit important to consider, we need one person to be in both pairs in the swap. So John must be in both pairs like $(($ John,Alex $),($ John,Mark $))$. This swap indicates that we're going to take a race where the pair $($ John,Alex $)$ exists and a race where $($ John,Mark $)$ exists and make the swap of Alex for Mark. The reason we want to do this is because, hypothetically, the pair $($ John,Alex $)$ appears far more often than the pair $($ John,Mark $)$, so we've like to increase the number of $($ John,Mark $)$ pairs in exchange for one of $($ John,Alex $)$. You should note, this should increase the overall evenness of all the races by definition. This is like a pseudo-gradient descent.

We order the set of all swaps by the difference between how many times they occur. For example, if $($ John,Alex $)$ occurs 10 times and $($ John,Mark $)$ occurs just once, the difference is 9. If $($ John,Joseph $)$ occurs only 3 times, then it would be best if our swap was $(($ John,Alex $),($ John,Mark $))$ because the difference is 9. We assign probabilities of taking each swap by just doing $\text{Pr}(\text{performing the swap})=\frac{\text{difference(swap)}}{\text{num total swaps possible}}$.

We do this a number of times (precisely `config.yaml/algorithmic/num_optimize_iterations` times) and then yield a set of races that have maximized the overall evenness fitness. However, you may have guessed, it's not a stricly concave/convex function, so there exist local minima/maxima that the program can get stuck in. i.e. there may be a swap that in the short term makes evenness worse, but in the long term can yield even better evenness than if we never made it in the first place. To combat this, we first make it possible for the algorithm to take a swap that may not have a large difference. And secondly, we allow the program to make swaps that actually do decrease the overall evenness fitness by using an _entropy_ formula. We start with an initial temperature, $t_0\in\mathbb{R}$ and cooling rate $c\in\mathbb{R}$ and for every iteration, $i\in[1..\text{num iterations}]$, the probability of taking a swap that decreases fitness is $e^{\Delta / t_i}$ where $\Delta=\text{fitness(new races)}-\text{fitness(old races)}$ and $t_i=\frac{t_0}{1+c\cdot i}$. This is called simulated annealing and allows the program to escape local minima and maxima to hopefully reach global minima/maxima for the most optimal set of races.

Once that process has completed, we end up with an unordered set of races that minimize the number of total races (maximum average $k$), exactly have $R$ for every person and maximizes the evenness fitness. Now a new auxilliary problem appears:
> **How do we order the races?** It matters. We could just throw all of John's races at the very start and Alex's at the very end, but that's super lame for both players. Because then they're just sitting there waiting and watching. However, it's fair! But we don't just care about fair here, we care about _fun_.

To combat this problem, almost the exact same thing above except we define a new _swapping fitness_ function that outputs a higher score if every person has their races spread out as much as possible while keeping it even across all the players. We perform the same process as above with its own initial temperature, cooling rate, and number of iterations. Except the swaps we make here are just the positions of the races (e.g. race 1 swapped for race 4).

Once this is done, we print out some statistics into the terminal, save the output, and display the results.

### Evenness fitness function
This function is a crucial part of this whole operation. It starts off by calculating if any race is invalid. If so, just return $-\infty$ immediately. If not, then we calculate how many times each pair appears in the list of all the races. The fitness is the negative variance of those values:

$\text{fitness(races)}=\left\{\begin{matrix}-\infty&\text{ if invalid}\\-\text{Variance}(\text{counts of all pairs})&\text{ otherwise}\end{matrix}\right.$

The idea behind this is if all the counts of all the pairs is the same number, the variance is 0. If we see a wider range of the counts of the pairs then the variance will be higher (and fitness lower). So the goal of the entire solver is then to move towards all the counts being the same number — which is exactly what we want.

### Swapping fitness function
Another crucial component of the second part of the entire algorithm. The goal in this function is two-fold:
- To make the distance between races for every player fair, and even
- To increase the distance between every race the player has

So the word _even_ should given a clue to you that we use variance again. However, this isn't enough this time, because technically if we have John perform 4 races all that the start (without Alex in any of them) and then Alex race all 4 races of his at the end (without John) then the variance of the space between each of their races is technically 0 — perfect! But not really what we want. Because like I mentioned previously, this isn't fun, even if it's fair. John doesn't want to smash 4 races down really fast and then wait the entire rest of his evening watching Mark and Alex race, he wants to race, break, then race again all the way to the end.

This is where we bring in the part about increasing the distance between all the races. To accomplish this, we try to maximize the mean distance between races.

We design a function that first notes down all the races that the players are apart of, by index. These appearances are sorted by index, and then we calculate the distance between all of the player's appearances from the last one. So if John raced in races `[1,2,5,7]` this would make the distances now be `[2-1,5-2,7-5] = [1,3,2]`. We then note down the average spacing for John, which in this case is `= (1+3+2)/3 = 6/3 = 2`. Then finally, we take all these average spacings for every player, calculate their mean and variance and make the fitness function just simply $\text{fitness(all races)}=\text{mean spacing}-\text{variance}$.

To recap, we calculate the mean spacing for every player. We want to maximize this overall. We want the mean spacing to be the same for every player, so we want to minimize variance. Voilà! Victory achieved.

## Application
I don't feel it entirely necessary to explain how the application works, but I'll give a few details if you're really curious.

### Score calculation
This is one I can maybe see being useful information. I asked myself a few questions:
- Is a player getting 1st place in a four person race worth more than a player getting 1st place in a three person race?
- Is a player getting last in a four person race worth less than a player getting last in a three person race?
- Is a player getting 1st place in a three person race worth more than getting 2nd in a four person race?
- And so on...

And determined that my answers were: yes, yes, and yes. So I chose to make the points awarded as follows:

| position | 4 people | 3 people |
|---|---|---|
1st | 24 | 20
2nd | 16 | 12
3rd | 8 | 4
4th | 0 | N/A

You might be asking yourself, why not just divide all those points by 4? To that I say, because big numbers are more fun than smaller numbers. I also like the numbers 24 and 20 more than 6 and 5 for a maximum score per race. Maybe it also has the advantage of shrouding the scoring in mystery as well, so players are less likely to start thinking really hard about how many points they need to get to win and just focus on having fun and seeing the numbers go up. Maybe that's not a good thing and instead I just confused and/or angered a bunch of drunken Mario Karters... oh well.

### Initializing from the output of the Bracket Generator
The Bracket Generator program saves its output into a `pickle`d file using the `pickle` python library. The application at the very start just unpickles those races and then determines the roster from everyone that appeared in the races.

It also reads in the `config.yaml` file to get the list of `previous_races` and `people_that_left` because we want to use those in the application. We want to use the `previous_races` to know where to start displaying the leaderboard from. We don't want to make the user have to input the placements of each race again (because they will not remember). We also want to cross out people on the scoreboard if they left because it's fun that way.

We initialize both the scoreboard and the current races portion with all this information, and roll the entire program forward using the information from `previous_races`.

### Being able to go back
For this, each component has two functions:
- `get_state` which returns all important information for reconstructing the component as is
- `set_state` which takes in a previous output from `get_state` and constructs itself according to that input.

So any time an action is performed within the application, like going forward to the next race, it just saves the current state of the program before moving forward in a stack. When you wish to go back, it pops from the stack and sets the state.

### Saving and Loading using S and L
The easiest part. We take every component, grab its state using `get_state` and use the `pickle` module to write it to disk. To load, we just load that state back and use `set_state`. To output the list of previous races, we just read from the current races component where we currently are and convert those races into a string and output them to the terminal. Is it convenient? No. But it's easy and quick.
