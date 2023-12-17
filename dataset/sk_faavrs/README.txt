Thank you for downloading FAAVRS:
Finally, An Alternative to VRoid Studio.

Current Version: 1.1
Version Release Date: 6/21/2023


Latest Patch's Patch Notes:


-Fixed a weight painting issue at the knees that caused some erroneous stretching when moving the legs (6/19/23)
-Improved vertex normals for the entire body[Dynamic shadows should now look better]
(6/20/23)



Table of Contents(TO FINISH):

Line 28: About FAAVRS
Line 53: FAQ
Line 86: Face Tracking Blendshapes
Line 103: Copyright/Other Information



About FAAVRS:

I got tired of having a crappy-looking VRoid avatar that had numerous technical problems,
didn't run very well, had LOADS of issues related to transparency, 
didn't support NSFW very well, the works. 
So, I began searching around for a new base I could build upon.

On my quest for a good-looking avatar I could call my own, 
I scoured Booth and Gumroad for what was available for free.
I also checked out Koikatsu Party export tools. 

To my chagrin, everything available was either:

-Incredibly expensive
-Locked under strict copyright in terms of asset redistribution
-Looked absolutely terrible
-Horribly unoptimized/awful weight painting/over-reliance on alpha transparency/other technical problems
-Lacked proper NSFW support
-Some combination of the above

So, I got to work on building my own foundation. 
After being stuck in development hell for like, 4 months before coming back to the project,
(college got in the way)
I am proud to announce that I've added enough features to this base to release it to the public.



FAQ:

Q: Is there any form of monetization attached? 
A: Nope! I don't even have a Patreon as of writing this.

Q: Why are all the textures pink?
A: It means there's no texture assigned to the material. 
Go to Texture Paint, (at the top) select the image icon(next to an image name) and select the mountain icon thingy.
If you don't see any images up there, click the folder icon(Open Image) and go to the textures folder.

The textures used in the current version are Character_Face5.png and Character_Body4.png, which should be in the latest version's folder.


Q: Where's CATS?
A: Grab it here: https://github.com/absolute-quantum/cats-blender-plugin/archive/refs/heads/development.zip
This is the Blender 3 version. 
It would have been on the GitHub had the repo's owner not ghosted his entire dev team.
Also, here's the plugin containing the baking feature: https://github.com/feilen/tuxedo-blender-plugin

Q: Where's the "Eyebrows Suprised" blendshape?
A: I felt no need to make one, just combine Eyebrows Happy with Eyebrows Rotate(Outer) in Unity animations.

Q: Pout emote?
A: I added Puff Cheeks back into the Mouth section just for this.

Q: Is the PP a mesh toggle?
A: Nope, I made it a shape key/blendshape for optimization reasons. 
If you want to edit the PP mesh, set Show PP to 1, click the nearby dropdown, 
and hit "Apply Selected Shapekey to Basis". It'll reverse the direction of "Show PP".




Face Tracking Blendshapes:

Based on VRCFaceTracking's Unified Expressions Blended Shapes: 

EyeClosed -> Eye Close
EyeWide ->  Eyes Suprised
EyeDilation -> Pupils Dilated

MouthSmile -> Smile 

The rest of these are located under the FACE TRACKING menu.

"LipPucker(Reduced) & LipFunnel(Reduced)?" 
I made those to be the same blendshape on purpose as like, combined.
Should reduce some issues you might be having if the normal blendshapes don't work right.


Copyright/Other Information:

"Genshin Style Anime Female Base Mesh For Blender" (https://skfb.ly/ooLoO) by David Onizaki is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
Heavy amounts of changes were made, from the textures to the rigging, weight painting, & shape keys,
as well as an entire new mesh for the PP.