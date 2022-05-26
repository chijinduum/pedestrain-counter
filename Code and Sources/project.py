from sourced.centroidtracker import CentroidTracker 
from sourced.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np 
import argparse
import imutils
import dlib
import cv2
import sqlite3

# create the argument parse and identify the arguments for the code
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", nargs='+', type=str,
	help="input video file(s)")
ap.add_argument("-t", "--table", type=str, default="data_report",
	help="name of table to store the data report in")
ap.add_argument("-o", "--output", type=str,
	help="output updated video file(s)")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# import mobileSSD files from folder
prototxt = "MobileNetSSD_deploy.prototxt.txt"
caffemodel = "MobileNetSSD_deploy.caffemodel"


# identify the classes the MobileNetSSD should detect in the video
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


# load imported SSD files  
print("[INFO] Loading SSD files...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# if there is no input file, exit code
if not args.get("input", False):
	exit()

print("[INFO] Running video file...")

# Import SQL into Python and create a database

con = sqlite3.connect('report.db')
cur = con.cursor()

# create a table that will store the infromation from the video
cur.execute('''
CREATE TABLE IF NOT EXISTS `%s` (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	footage_name VARCHAR(75) NOT NULL,
	no_up INTEGER NOT NULL,
	no_down INTEGER NOT NULL,
	total_no INTEGER NOT NULL
);''' % (args["table"]))

# Print into created database
con.commit()


# Input the video file into CV
for video_source in args["input"]:
	vs = cv2.VideoCapture(video_source)

	# Initialize the video writer and frame dimenstions (width and height)
	writer = None
	W = None
	H = None

	# Initiatize the tracker from the sourced centroid tracker file
	track = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackable_Objects = {}


	# Initialize the total number of frames processed so far, the number of people moving up or moving down
	total_frames = 0
	going_down = 0
	going_up = 0

	# Start the frames per second throughput estimator
	fps = FPS().start()

	# LOOPING THROUGH THE FRAMES OF THE VIDEO STREAM
	while True:

		frame = vs.read()
		frame = frame [1]

		# If we move through the video and there are no more frames to process,then end the program
		if video_source is not None and frame is None:
			break


		# Resize the frame to have a maximum width and height	 of 1500 pixels( for faster processing, then convert to RGB for dlib
		frame = imutils.resize(frame, width=1500, height=1500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Set the frame dimensions if there are none
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# Initializing the writer to write the video
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W,H), True)

		status = "Waiting"
		rects = []

		# Checking to see if a more computationally detection method can aid the tracker
		if total_frames % args["skip_frames"] == 0:


			# Set the status and initialize a new set of object trackers
			status = "Detecting"
			trackers = []


			# Convert the frame to a blob and pass it through a network to detect people
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()


			# LOOP THROUGH THE DETECTIONS
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]

				# Filter out detections below the set confidence
				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])

					# If it is not a person, ignore it
					if CLASSES[idx] != "person":
						continue


					# Compute the x and y coordinates of the bounding box	
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# Construct a dlib rectangle object from the bounding box coordinates and then start the dlib coordination trackers
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# Add the tracker to our lost of trackers
					trackers.append(tracker)


		# Use object trackers instead of object detectors. This is forhigher frame processing
		else:
			# LOOP OVER THE TRACKERS
			for tracker in trackers:

				# Set to tracking rather than waiting or detecting
				status = "Tracking"


				# Update the tracker and grab the position
				tracker.update(rgb)
				position = tracker.get_position()

				# Unpack the position object and add the bounding box coordinates to the rectangle's list
				startX = int(position.left())
				startY = int(position.top())
				endX = int(position.right())
				endY = int(position.bottom())


				
				rects.append((startX, startY, endX, endY))


		# Draw a horizontal line in the center of the frame to determine whether a person is moving up or down
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)


		# Use the tracker to associate old objects with objects that have been computed newly
		objects = track.update(rects)

		# LOOP OVER THE TRACKED OBJECTS
		for (objectID, centroid) in objects.items():

			# Check to see if a trackable object exists for the current object ID
			to = trackable_Objects.get(objectID, None)

			# If there is no existing trackable object, then create one
			if to is None: 
				to = TrackableObject(objectID, centroid)

			# Else there is a trackable object that can be used to detemine the direction	
			else:


				# The difference between the y-coordinate and the mean of the previous centroids will determine the direction of the moving object

				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# To check of the object has been counted or not
				if not to.counted:
					# If the direction is negative and the centroidis above the line, it is going up
					if direction < 0 and centroid[1] < H //2:
						going_up += 1
						to.counted = True

					# If the direction is positive and the centroidis below the line, it is going down
					elif direction > 0 and centroid[1] > H //2:
						going_down += 1
						to.counted = True

			#Store the object in the dictionary
			trackable_Objects[objectID] = to

			#Draw both the ID of the object and the centroid of the object on the output frame
			text = "Person {}". format(objectID)
			cv2.putText (frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# Construct information that will display on the video
		info = [
			("Going Up", going_up),
			("Going Down", going_down),
			("Status", status)
		]

		# Loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


		# Check to see writing in the image frame is needed
		if writer is not None:
			writer.write(frame)


		# Output the image frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# Pressing 'B' to break the loop if necessary
		if key == ord("b"):
			break

		# Increment the total number of frames processed snd update the counter
		total_frames += 1
		fps.update()

	# Stop the timer and display estimated FPS information
	fps.stop()
	print("[INFO] File run successful...")
	print("[INFO] Time elapsed: {:.2f}...".format(fps.elapsed()))
	print("[INFO] Estimated FPS: {:.2f}...".format(fps.fps()))

	# Check to see if the video writer pointer is released
	if writer is not None:
		writer.release()

	# Stop the video if the file is not going to be used
	if not args.get("input", False):
		vs.stop()

	# Otherwise, release the output file
	else:
		vs.release()


	# Add the results to the database
	cur.execute('''INSERT INTO `%s` (footage_name, no_up, no_down, total_no) VALUES ("%s", %d, %d, %d)'''
		% (args["table"], video_source, going_up, going_down, going_up + going_down))

	con.commit()

	#Close any open windows
	cv2.destroyAllWindows()
# Close the database
con.close()		