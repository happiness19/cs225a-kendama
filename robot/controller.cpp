/**
 * @file controller.cpp
 * @brief Controller file
 *
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <algorithm>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
bool runloop = false;
void sighandler(int) { runloop = false; }

#include "redis_keys.h"

int main()
{
	// Location of URDF files specifying world and robot information
	static const string robot_file = string(CS225A_URDF_FOLDER) + "/flexiv/flexiv.urdf";

	// start redis client
	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots, read current state and update the model
	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
	robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
	robot->updateModel();

	// prepare controller
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// arm task
	const string control_link = "link7";
	const Vector3d control_point = Vector3d(0, 0, 0.081);
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = control_point;
	auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, compliant_frame);
	pose_task->setPosControlGains(1600, 90, 0);
	pose_task->setOriControlGains(700, 55, 0);

	const Vector3d ee_pos_initial = robot->position(control_link, control_point);
	const Matrix3d ee_ori_initial = robot->rotation(control_link);
	const Vector3d ee_pos_goal = ee_pos_initial + Vector3d(0.0, 0.0, 0.5);

	// joint task
	auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	joint_task->setGains(400, 40, 0);

	VectorXd q_desired(dof);
	q_desired << robot->q(); // set desired joint angles same as the initial sim configuration
	joint_task->setGoalPosition(q_desired);

	// create a loop timer
	runloop = true;
	double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (runloop)
	{
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();

		// update robot
		robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
		robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
		robot->updateModel();

		// Move quickly upward, then hold the goal at about 0.5 m.
		const double motion_duration = 0.18;
		const double s = std::min(time / motion_duration, 1.0);
		const double smooth_step = s * s * (3.0 - 2.0 * s);
		const Vector3d ee_pos_desired = ee_pos_initial + smooth_step * (ee_pos_goal - ee_pos_initial);
		pose_task->setGoalPosition(ee_pos_desired);
		pose_task->setGoalOrientation(ee_ori_initial);

		// update task model
		N_prec.setIdentity();
		pose_task->updateTaskModel(N_prec);
		joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

		command_torques = pose_task->computeTorques() + joint_task->computeTorques();

		// execute redis write callback
		redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	}

	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * command_torques); // back to floating

	return 0;
}
