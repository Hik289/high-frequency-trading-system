package com.boot.service;

import java.util.List;

import com.boot.entity.Team;
import com.boot.entity.User;

public interface TeamService {

	List<Team> getList();

	void save(Team team);

	Team getDeatil(Integer teamId);

	List<User> getTeamUser(Integer teamId);

	void addTeamUser(Integer teamId, Integer userId);

	Team getTeamDefault(Integer userId);

	void removeUser(Integer teamId, Integer userId);

}
