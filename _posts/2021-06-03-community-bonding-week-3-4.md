---
layout: page
title: git rebase, git commit, git teach me how to contribute to open source
date: 2021-06-03 12:00:00-0400
description: Community Bonding Weeks 3 and 4
---

The rest of community bonding was spent interacting with the PyMC developpers and, above all, learning how to better use GitHub. On the statistical theory side, things are progressing faster. Admittedly, on the coding side, I have come to full realization that my foundation is more lacking and there is a lot to learn.

Emotionally, it has sometimes been difficult to accept my slow progress and feeling completely lost… I feel that I am currently in a slump and am not too sure where to even look for directions. There are many weeks left, so lots of room to better myself!

Here are some lessons and some areas that I need to work on for the coming weeks.

## 1 - Managing a better Git workflow

I will be working on different branches at a time. For instance, my implementation of a DP submodule should be done independently from addressing PyMC3 issues. Familiarizing myself with how to use git merge, git reset and git rebase will be very important to ensure properly flow as as aspiring developper. For now, what I have gathered is summarized in the following steps as I am working on my branch dp-gsoc:

```bash
git commit my progress. # I don’t have to push to origin yet!
git checkout main to go back to main and
git pull upstream to update my local repository with respect to upstream. It would be also nice to push to origin main with a simple git push -u origin main.
git checkout dp-gsoc followed by
git merge main to update my local branch with the updates from upstream main
```

This is a failed-proof method since I am currently working on a brand new submodule, so I don’t expect to encounter any merge conflicts 😅

2 - Asking questions
Asking the right and right number of questions has also not been easy. I can spend countless hours asking PyMC developpers and contributors for pointers and answers, but, in many instances, I just often have a feeling that my questions are too “basic”. This “imposter syndrome” feeling is not fun, but I try very hard to go through all (if not most) of the documentation before asking any questions. However, if I’m stuck for any longer, I feel that I should not hesitate.

3 - Knowing what I don’t need to know
When it came to understanding the codebase, it was initially difficult for me to discern what I need to know for my project versus what I don’t need to know. For instance, PyMC3 is built on top aesara which, apparently, not everybody understands to its full extent. And that’s more than okay to keep working with my GSoC project :')