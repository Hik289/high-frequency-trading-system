package com.boot.service.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.boot.dao.TeamDao;
import com.boot.entity.Team;
import com.boot.entity.User;
import com.boot.repository.TeamRepository;
import com.boot.service.TeamService;
import com.boot.vo.TokenHandler;

@Service
public class TeamServiceImpl implements TeamService {
	@Autowired
	private TeamDao teamDao;
	@Autowired
	private TeamRepository teamRepository;

	@Override
	public List<Team> getList() {
		Integer userId = TokenHandler.getBusinessId();
		return teamDao.getList(userId);
	}

	@Override
	@Transactional(propagation=Propagation.REQUIRED,rollbackFor=Exception.class)
	public void save(Team team) {
		if(team.getId()== null) {
			Integer userId = TokenHandler.getBusinessId();
			team.setUser(userId);
		}
		teamRepository.save(team);
	}

	@Override
	public Team getDeatil(Integer teamId) {
		return teamDao.getDeatil(teamId);
	}

	@Override
	public List<User> getTeamUser(Integer teamId) {
		return teamDao.getTeamUser(teamId);
	}

	@Override
	public void addTeamUser(Integer teamId, Integer userId) {
		Integer count = teamDao.countByTeamUser(teamId,userId);
		if(count<=0) {
			teamDao.insertTeamUser(teamId,userId);
		}
	}

	@Override
	public Team getTeamDefault(Integer userId) {
		return teamDao.getTeamDefault(userId);
	}

	@Override
	public void removeUser(Integer teamId, Integer userId) {
		teamDao.removeUser(teamId,userId);
	}
}
